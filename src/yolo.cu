#include "infer.hpp"
#include "yolo.hpp"

namespace yolo {

using namespace std;

#define GPU_BLOCK_THREADS 512
#define checkRuntime(call)                                                                 \
  do {                                                                                     \
    auto ___call__ret_code__ = (call);                                                     \
    if (___call__ret_code__ != cudaSuccess) {                                              \
      INFO("CUDA Runtime error💥 %s # %s, code = %s [ %d ]", #call,                         \
           cudaGetErrorString(___call__ret_code__), cudaGetErrorName(___call__ret_code__), \
           ___call__ret_code__);                                                           \
      abort();                                                                             \
    }                                                                                      \
  } while (0)

#define checkKernel(...)                 \
  do {                                   \
    { (__VA_ARGS__); }                   \
    checkRuntime(cudaPeekAtLastError()); \
  } while (0)

enum class NormType : int { None = 0, MeanStd = 1, AlphaBeta = 2 };

enum class ChannelType : int { None = 0, SwapRB = 1 };

/* 归一化操作，可以支持均值标准差，alpha beta，和swap RB */
struct Norm {
  float mean[3];
  float std[3];
  float alpha, beta;
  NormType type = NormType::None;
  ChannelType channel_type = ChannelType::None;

  // out = (x * alpha - mean) / std
  static Norm mean_std(const float mean[3], const float std[3], float alpha = 1 / 255.0f,
                       ChannelType channel_type = ChannelType::None);

  // out = x * alpha + beta
  static Norm alpha_beta(float alpha, float beta = 0, ChannelType channel_type = ChannelType::None);

  // None
  static Norm None();
};

Norm Norm::mean_std(const float mean[3], const float std[3], float alpha,
                    ChannelType channel_type) {
  Norm out;
  out.type = NormType::MeanStd;
  out.alpha = alpha;
  out.channel_type = channel_type;
  memcpy(out.mean, mean, sizeof(out.mean));
  memcpy(out.std, std, sizeof(out.std));
  return out;
}

Norm Norm::alpha_beta(float alpha, float beta, ChannelType channel_type) {
  Norm out;
  out.type = NormType::AlphaBeta;
  out.alpha = alpha;
  out.beta = beta;
  out.channel_type = channel_type;
  return out;
}

Norm Norm::None() { return Norm(); }

const int NUM_BOX_ELEMENT = 8;  // left, top, right, bottom, confidence, class,
                                // keepflag, row_index(output)
const int MAX_IMAGE_BOXES = 1024;
inline int upbound(int n, int align = 32) { return (n + align - 1) / align * align; }
static __host__ __device__ void affine_project(float *matrix, float x, float y, float *ox,
                                               float *oy) {
  *ox = matrix[0] * x + matrix[1] * y + matrix[2];
  *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __global__ void decode_kernel_common(float *predict, int num_bboxes, int num_classes,
                                            int output_cdim, float confidence_threshold,
                                            float *invert_affine_matrix, float *parray,
                                            int MAX_IMAGE_BOXES) {
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  if (position >= num_bboxes) return;

  float *pitem = predict + output_cdim * position;
  float objectness = pitem[4];
  if (objectness < confidence_threshold) return;

  float *class_confidence = pitem + 5;
  float confidence = *class_confidence++;
  int label = 0;
  for (int i = 1; i < num_classes; ++i, ++class_confidence) {
    if (*class_confidence > confidence) {
      confidence = *class_confidence;
      label = i;
    }
  }

  confidence *= objectness;
  if (confidence < confidence_threshold) return;

  int index = atomicAdd(parray, 1);
  if (index >= MAX_IMAGE_BOXES) return;

  float cx = *pitem++;
  float cy = *pitem++;
  float width = *pitem++;
  float height = *pitem++;
  float left = cx - width * 0.5f;
  float top = cy - height * 0.5f;
  float right = cx + width * 0.5f;
  float bottom = cy + height * 0.5f;
  affine_project(invert_affine_matrix, left, top, &left, &top);
  affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

  float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
  *pout_item++ = left;
  *pout_item++ = top;
  *pout_item++ = right;
  *pout_item++ = bottom;
  *pout_item++ = confidence;
  *pout_item++ = label;
  *pout_item++ = 1;  // 1 = keep, 0 = ignore
}

static __global__ void decode_kernel_v8(float *predict, int num_bboxes, int num_classes,
                                        int output_cdim, float confidence_threshold,
                                        float *invert_affine_matrix, float *parray,
                                        int MAX_IMAGE_BOXES) {
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  if (position >= num_bboxes) return;

  float *pitem = predict + output_cdim * position;
  float *class_confidence = pitem + 4;
  float confidence = *class_confidence++;
  int label = 0;
  for (int i = 1; i < num_classes; ++i, ++class_confidence) {
    if (*class_confidence > confidence) {
      confidence = *class_confidence;
      label = i;
    }
  }
  if (confidence < confidence_threshold) return;

  int index = atomicAdd(parray, 1);
  if (index >= MAX_IMAGE_BOXES) return;

  float cx = *pitem++;
  float cy = *pitem++;
  float width = *pitem++;
  float height = *pitem++;
  float left = cx - width * 0.5f;
  float top = cy - height * 0.5f;
  float right = cx + width * 0.5f;
  float bottom = cy + height * 0.5f;
  affine_project(invert_affine_matrix, left, top, &left, &top);
  affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

  float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
  *pout_item++ = left;
  *pout_item++ = top;
  *pout_item++ = right;
  *pout_item++ = bottom;
  *pout_item++ = confidence;
  *pout_item++ = label;
  *pout_item++ = 1;  // 1 = keep, 0 = ignore
  *pout_item++ = position;
}

static __device__ float box_iou(float aleft, float atop, float aright, float abottom, float bleft,
                                float btop, float bright, float bbottom) {
  float cleft = max(aleft, bleft);
  float ctop = max(atop, btop);
  float cright = min(aright, bright);
  float cbottom = min(abottom, bbottom);

  float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
  if (c_area == 0.0f) return 0.0f;

  float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
  float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
  return c_area / (a_area + b_area - c_area);
}

static __global__ void fast_nms_kernel(float *bboxes, int MAX_IMAGE_BOXES, float threshold) {
  int position = (blockDim.x * blockIdx.x + threadIdx.x);
  int count = min((int)*bboxes, MAX_IMAGE_BOXES);
  if (position >= count) return;

  // left, top, right, bottom, confidence, class, keepflag
  float *pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
  for (int i = 0; i < count; ++i) {
    float *pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
    if (i == position || pcurrent[5] != pitem[5]) continue;

    if (pitem[4] >= pcurrent[4]) {
      if (pitem[4] == pcurrent[4] && i < position) continue;

      float iou = box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1],
                          pitem[2], pitem[3]);

      if (iou > threshold) {
        pcurrent[6] = 0;  // 1=keep, 0=ignore
        return;
      }
    }
  }
}

static dim3 grid_dims(int numJobs) {
  int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
  return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
}

static dim3 block_dims(int numJobs) {
  return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
}

static void decode_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                  float confidence_threshold, float nms_threshold,
                                  float *invert_affine_matrix, float *parray, int MAX_IMAGE_BOXES,
                                  Type type, cudaStream_t stream) {
  auto grid = grid_dims(num_bboxes);
  auto block = block_dims(num_bboxes);

  if (type == Type::V8 || type == Type::V8Seg) {
    checkKernel(decode_kernel_v8<<<grid, block, 0, stream>>>(
        predict, num_bboxes, num_classes, output_cdim, confidence_threshold, invert_affine_matrix,
        parray, MAX_IMAGE_BOXES));
  } else {
    checkKernel(decode_kernel_common<<<grid, block, 0, stream>>>(
        predict, num_bboxes, num_classes, output_cdim, confidence_threshold, invert_affine_matrix,
        parray, MAX_IMAGE_BOXES));
  }

  grid = grid_dims(MAX_IMAGE_BOXES);
  block = block_dims(MAX_IMAGE_BOXES);
  checkKernel(fast_nms_kernel<<<grid, block, 0, stream>>>(parray, MAX_IMAGE_BOXES, nms_threshold));
}

static __global__ void warp_affine_bilinear_and_normalize_plane_kernel(
    uint8_t *src, int src_line_size, int src_width, int src_height, float *dst, int dst_width,
    int dst_height, uint8_t const_value_st, float *warp_affine_matrix_2_3, Norm norm) {
  int dx = blockDim.x * blockIdx.x + threadIdx.x;
  int dy = blockDim.y * blockIdx.y + threadIdx.y;
  if (dx >= dst_width || dy >= dst_height) return;

  float m_x1 = warp_affine_matrix_2_3[0];
  float m_y1 = warp_affine_matrix_2_3[1];
  float m_z1 = warp_affine_matrix_2_3[2];
  float m_x2 = warp_affine_matrix_2_3[3];
  float m_y2 = warp_affine_matrix_2_3[4];
  float m_z2 = warp_affine_matrix_2_3[5];

  float src_x = m_x1 * dx + m_y1 * dy + m_z1;
  float src_y = m_x2 * dx + m_y2 * dy + m_z2;
  float c0, c1, c2;

  if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
    // out of range
    c0 = const_value_st;
    c1 = const_value_st;
    c2 = const_value_st;
  } else {
    int y_low = floorf(src_y);
    int x_low = floorf(src_x);
    int y_high = y_low + 1;
    int x_high = x_low + 1;

    uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
    float ly = src_y - y_low;
    float lx = src_x - x_low;
    float hy = 1 - ly;
    float hx = 1 - lx;
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    uint8_t *v1 = const_value;
    uint8_t *v2 = const_value;
    uint8_t *v3 = const_value;
    uint8_t *v4 = const_value;
    if (y_low >= 0) {
      if (x_low >= 0) v1 = src + y_low * src_line_size + x_low * 3;

      if (x_high < src_width) v2 = src + y_low * src_line_size + x_high * 3;
    }

    if (y_high < src_height) {
      if (x_low >= 0) v3 = src + y_high * src_line_size + x_low * 3;

      if (x_high < src_width) v4 = src + y_high * src_line_size + x_high * 3;
    }

    // same to opencv
    c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
    c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
    c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
  }

  if (norm.channel_type == ChannelType::SwapRB) {
    float t = c2;
    c2 = c0;
    c0 = t;
  }

  if (norm.type == NormType::MeanStd) {
    c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
    c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
    c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
  } else if (norm.type == NormType::AlphaBeta) {
    c0 = c0 * norm.alpha + norm.beta;
    c1 = c1 * norm.alpha + norm.beta;
    c2 = c2 * norm.alpha + norm.beta;
  }

  int area = dst_width * dst_height;
  float *pdst_c0 = dst + dy * dst_width + dx;
  float *pdst_c1 = pdst_c0 + area;
  float *pdst_c2 = pdst_c1 + area;
  *pdst_c0 = c0;
  *pdst_c1 = c1;
  *pdst_c2 = c2;
}

static void warp_affine_bilinear_and_normalize_plane(uint8_t *src, int src_line_size, int src_width,
                                                     int src_height, float *dst, int dst_width,
                                                     int dst_height, float *matrix_2_3,
                                                     uint8_t const_value, const Norm &norm,
                                                     cudaStream_t stream) {
  dim3 grid((dst_width + 31) / 32, (dst_height + 31) / 32);
  dim3 block(32, 32);

  checkKernel(warp_affine_bilinear_and_normalize_plane_kernel<<<grid, block, 0, stream>>>(
      src, src_line_size, src_width, src_height, dst, dst_width, dst_height, const_value,
      matrix_2_3, norm));
}

//用于解码单个掩码（mask）
/*
什么叫解码单个掩码？

解码单个掩码是指从模型的输出中还原出单个实例的掩码。
在目标检测和实例分割任务中，模型通常会输出每个检测实例的信息，包括其边界框（Bounding Box）位置和对应的掩码。

掩码是用于描述目标实例的像素级别的信息，通常用于实例分割任务。
在深度学习模型中，这些掩码通常以概率图或二进制图的形式出现，用于指示图像中每个像素属于目标实例的概率或二进制状态。

解码单个掩码的过程就是将模型输出的信息转换为可视化或应用所需的形式。
在下面CUDA核函数中，
解码的过程包括将预测的掩码与权重相乘并应用 Sigmoid 函数，
最终将得到的值映射到 0 到 255 的范围，形成最终的可视化掩码。
这个掩码可以用于可视化目标实例的位置和形状。
*/
static __global__ void decode_single_mask_kernel(int left, int top, float *mask_weights,
                                                 float *mask_predict, int mask_width,
                                                 int mask_height, unsigned char *mask_out,
                                                 int mask_dim, int out_width, int out_height) {
  // mask_predict to mask_out
  // mask_weights @ mask_predict
  int dx = blockDim.x * blockIdx.x + threadIdx.x;
  int dy = blockDim.y * blockIdx.y + threadIdx.y;
  if (dx >= out_width || dy >= out_height) return;

  int sx = left + dx;
  int sy = top + dy;
  if (sx < 0 || sx >= mask_width || sy < 0 || sy >= mask_height) {
    mask_out[dy * out_width + dx] = 0;
    return;
  }

  float cumprod = 0;
  for (int ic = 0; ic < mask_dim; ++ic) {
    float cval = mask_predict[(ic * mask_height + sy) * mask_width + sx];
    float wval = mask_weights[ic];
    cumprod += cval * wval;
  }

  float alpha = 1.0f / (1.0f + exp(-cumprod));
  mask_out[dy * out_width + dx] = alpha * 255;
}

static void decode_single_mask(float left, float top, float *mask_weights, float *mask_predict,
                               int mask_width, int mask_height, unsigned char *mask_out,
                               int mask_dim, int out_width, int out_height, cudaStream_t stream) {
  // mask_weights is mask_dim(32 element) gpu pointer
  dim3 grid((out_width + 31) / 32, (out_height + 31) / 32);
  dim3 block(32, 32);

  checkKernel(decode_single_mask_kernel<<<grid, block, 0, stream>>>(
      left, top, mask_weights, mask_predict, mask_width, mask_height, mask_out, mask_dim, out_width,
      out_height));
}

const char *type_name(Type type) {
  switch (type) {
    case Type::V5:
      return "YoloV5";
    case Type::V3:
      return "YoloV3";
    case Type::V7:
      return "YoloV7";
    case Type::X:
      return "YoloX";
    case Type::V8:
      return "YoloV8";
    default:
      return "Unknow";
  }
}

/*
仿射变换矩阵每个元素的含义
| a   b   c |
| d   e   f |

其中：
a 和 e 控制水平和垂直方向的缩放变换
b 和 d 控制水平和垂直方向的切变 （shear）变换 （切变变换是指在平行于坐标轴的方向上进行的拉伸或挤压操作）
c 和 f 控制水平和垂直方向的平移变换

对于二维平面上的点 （x, y）经过仿射变换后的新坐标(x', y') 的计算方法：
x' = ax + by + c
y' = dx + ey + f

*/

struct AffineMatrix {
  float i2d[6];  // image to dst(network), 2x3 matrix 表示图像到目标网络的仿射变换矩阵
  float d2i[6];  // dst to image, 2x3 matrix  表示目标网络到图像的仿射变换矩阵

  /*
  参数 from： 表示原图像的尺寸
  参数 to：   表示目标网络的尺寸
  该函数用来计算 i2d 和 d2i
  */
  void compute(const std::tuple<int, int> &from, const std::tuple<int, int> &to) {
    //计算缩放比例，使得目标尺寸使用源图像尺寸
    float scale_x = get<0>(to) / (float)get<0>(from);
    float scale_y = get<1>(to) / (float)get<1>(from);
    float scale = std::min(scale_x, scale_y);

    i2d[0] = scale;
    i2d[1] = 0;
    i2d[2] = -scale * get<0>(from) * 0.5 + get<0>(to) * 0.5 + scale * 0.5 - 0.5;
    i2d[3] = 0;
    i2d[4] = scale;
    i2d[5] = -scale * get<1>(from) * 0.5 + get<1>(to) * 0.5 + scale * 0.5 - 0.5;

    //计算目标到图像的仿射变换矩阵
    double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
    D = D != 0. ? double(1.) / D : double(0.);
    double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D, A21 = -i2d[3] * D;
    double b1 = -A11 * i2d[2] - A12 * i2d[5];
    double b2 = -A21 * i2d[2] - A22 * i2d[5];

    d2i[0] = A11;
    d2i[1] = A12;
    d2i[2] = b1;
    d2i[3] = A21;
    d2i[4] = A22;
    d2i[5] = b2;
  }
};

InstanceSegmentMap::InstanceSegmentMap(int width, int height) {
  this->width = width;
  this->height = height;
  checkRuntime(cudaMallocHost(&this->data, width * height));
}

InstanceSegmentMap::~InstanceSegmentMap() {
  if (this->data) {
    checkRuntime(cudaFreeHost(this->data));
    this->data = nullptr;
  }
  this->width = 0;
  this->height = 0;
}

class InferImpl : public Infer {
 public:
  shared_ptr<trt::Infer> trt_; // TensorRT 的推理类
  string engine_file_; // 存储
  Type type_; //模型类型
  float confidence_threshold_; //目标检测的置信度
  /*
  非最大抑制技术：用于去除冗余检测结果。对所有检测结果按照置信度进行排序，并选择具有最高置信度的检测结果为最终输出
  */
  float nms_threshold_;  //非最大抑制阈值 

  vector<shared_ptr<trt::Memory<unsigned char>>> preprocess_buffers_; //预处理缓冲区集合
  trt::Memory<float> input_buffer_, bbox_predict_, output_boxarray_; //存储GPU上的内存，分别存储 输入的图像数据 、 边界框的预测结果、推理工程中的输出信息
  trt::Memory<float> segment_predict_;  //存储深度学习模型对分割的预测结果 
  int network_input_width_, network_input_height_; //网络输入的宽度和高度
  
  /*
  归一化：用于将输入数据按照一定规则进行缩放，使其具有特定的分布范围。有助于提高模型的训练稳定性和收敛速度
  */
  Norm normalize_;  //归一化参数
  vector<int> bbox_head_dims_; //边界框头维度
  vector<int> segment_head_dims_; //分割头维度
  int num_classes_ = 0;  //类别数
  bool has_segment_ = false; //标志是否具有分割头
  bool isdynamic_model_ = false; //标志模型是否具有动态维度
  vector<shared_ptr<trt::Memory<unsigned char>>> box_segment_cache_; //用于存储分割头缓存的集合

  virtual ~InferImpl() = default;

  void adjust_memory(int batch_size) {
    // the inference batch_size
    size_t input_numel = network_input_width_ * network_input_height_ * 3;
    //为输入缓冲区分配GPU内存，大小为 batch_size * 网络宽度 * 网络高度 * 3
    input_buffer_.gpu(batch_size * input_numel);
    //为边界框预测结果缓冲分配GPU内存
    bbox_predict_.gpu(batch_size * bbox_head_dims_[1] * bbox_head_dims_[2]);
    // 为输出边界框数组缓冲区分配GPU内存
    output_boxarray_.gpu(batch_size * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));
    // 为输出边界框数组缓冲区分配GPU内存
    output_boxarray_.cpu(batch_size * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));

    if (has_segment_) //如果模型有分割头
      //为分割预测结果缓冲区分配GPU内存
      segment_predict_.gpu(batch_size * segment_head_dims_[1] * segment_head_dims_[2] *
                           segment_head_dims_[3]);

    // 如果预处理缓冲区的数量小于 batch_size
    if ((int)preprocess_buffers_.size() < batch_size) {
      for (int i = preprocess_buffers_.size(); i < batch_size; ++i)
        // 将缺少的预处理缓冲区添加到 preprocess_buffers_
        preprocess_buffers_.push_back(make_shared<trt::Memory<unsigned char>>());
    }
  }

  void preprocess(int ibatch, const Image &image,
                  shared_ptr<trt::Memory<unsigned char>> preprocess_buffer, AffineMatrix &affine,
                  void *stream = nullptr) {

    //计算仿射变换矩阵，将输入图像变换到模型期望的输入尺寸
    /* 
    仿射变换是一种线性变换，用于将二维平面上的图像进行平移、旋转、缩放和错切等操作。
    在仿射变换中，原始图像上的平行线仍然保持平行，不同点之间的距离比例保持不变。
    这种变换可以通过一个矩阵来表示，被称为仿射变换矩阵。
    */
    affine.compute(make_tuple(image.width, image.height),
                   make_tuple(network_input_width_, network_input_height_));

    size_t input_numel = network_input_width_ * network_input_height_ * 3;

    // 获取输入缓冲区的 GPU 指针，移动到当前 batch 的位置
    float *input_device = input_buffer_.gpu() + ibatch * input_numel;
    // 计算图像数据的大小
    size_t size_image = image.width * image.height * 3;

    // 计算仿射变换矩阵占用的空间大小，向上取整到 32 字节的倍数
    size_t size_matrix = upbound(sizeof(affine.d2i), 32);

    // 在预处理缓冲区中获取 GPU 指针，用于存储仿射变换矩阵和图像数据
    uint8_t *gpu_workspace = preprocess_buffer->gpu(size_matrix + size_image);

    // 获取仿射变换矩阵的 GPU 指针
    float *affine_matrix_device = (float *)gpu_workspace;

    // 获取图像数据的 GPU 指针
    uint8_t *image_device = gpu_workspace + size_matrix;

    // 在预处理缓冲区中获取 CPU 指针，用于存储仿射变换矩阵和图像数据的主机端数据
    uint8_t *cpu_workspace = preprocess_buffer->cpu(size_matrix + size_image);
    
    // 获取仿射变换矩阵的主机端指针
    float *affine_matrix_host = (float *)cpu_workspace;

    // 获取图像数据的主机端指针
    uint8_t *image_host = cpu_workspace + size_matrix;

    // speed up
    // 将输入图像数据和仿射变换矩阵数据从主机端复制到 GPU 端
    // 这里使用异步的方式进行数据传输，加速处理过程
    // 注意：这里使用了 CUDA 的 memcpy 函数和 cudaMemcpyAsync 函数
    cudaStream_t stream_ = (cudaStream_t)stream;
    memcpy(image_host, image.bgrptr, size_image);
    memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
    checkRuntime(
        cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
    checkRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i),
                                 cudaMemcpyHostToDevice, stream_));

    // 执行仿射变换、双线性插值和归一化操作
    // 这是对输入图像进行预处理的关键步骤
    /*
    双线性插值是一种在离散的数据点之间估算数值的方法，通常用于图像处理中的缩放和变换操作。
    在图像处理中，经常需要在图像的像素之间进行插值以获得非整数坐标处的像素值。
    双线性插值是一种简单而有效的插值方法，它利用了图像中相邻像素之间的局部线性关系
    */
    warp_affine_bilinear_and_normalize_plane(image_device, image.width * 3, image.width,
                                             image.height, input_device, network_input_width_,
                                             network_input_height_, affine_matrix_device, 114,
                                             normalize_, stream_);
  }

  bool load(const string &engine_file, Type type, float confidence_threshold, float nms_threshold) {
    trt_ = trt::load(engine_file);
    if (trt_ == nullptr) return false;

    trt_->print();

    this->type_ = type;
    this->confidence_threshold_ = confidence_threshold;
    this->nms_threshold_ = nms_threshold;

    auto input_dim = trt_->static_dims(0);
    bbox_head_dims_ = trt_->static_dims(1);
    has_segment_ = type == Type::V8Seg;
    if (has_segment_) {
      bbox_head_dims_ = trt_->static_dims(2);
      segment_head_dims_ = trt_->static_dims(1);
    }
    network_input_width_ = input_dim[3];
    network_input_height_ = input_dim[2];
    isdynamic_model_ = trt_->has_dynamic_dim();

    if (type == Type::V5 || type == Type::V3 || type == Type::V7) {
      normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
      num_classes_ = bbox_head_dims_[2] - 5;
    } else if (type == Type::V8) {
      normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
      num_classes_ = bbox_head_dims_[2] - 4;
    } else if (type == Type::V8Seg) {
      normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
      num_classes_ = bbox_head_dims_[2] - 4 - segment_head_dims_[1];
    } else if (type == Type::X) {
      // float mean[] = {0.485, 0.456, 0.406};
      // float std[]  = {0.229, 0.224, 0.225};
      // normalize_ = Norm::mean_std(mean, std, 1/255.0f, ChannelType::SwapRB);
      normalize_ = Norm::None();
      num_classes_ = bbox_head_dims_[2] - 5;
    } else {
      INFO("Unsupport type %d", type);
    }
    return true;
  }

  virtual BoxArray forward(const Image &image, void *stream = nullptr) override {
    auto output = forwards({image}, stream);
    if (output.empty()) return {};
    return output[0];
  }

  virtual vector<BoxArray> forwards(const vector<Image> &images, void *stream = nullptr) override {
    int num_image = images.size();
    if (num_image == 0) return {};

    /*
    获取模型输入的各个维度的大小。通畅包括
    1. 批处理大小（batch size）
    2. 通道数
    3. 图像高度
    4. 图像宽度
    */
    auto input_dims = trt_->static_dims(0);

    /*
    获取推断批处理大小
    批处理大小是指在模型训练或推理中一次输入给模型的样本数量。
    */
    int infer_batch_size = input_dims[0];
    if (infer_batch_size != num_image) { //批处理大小与输入图像数量不相等
      if (isdynamic_model_) { //模型为动态形状，允许在推理时更改输入的批处理大小
        infer_batch_size = num_image; 
        input_dims[0] = num_image;
        if (!trt_->set_run_dims(0, input_dims)) return {};
      } else {
        if (infer_batch_size < num_image) {
          INFO(
              "When using static shape model, number of images[%d] must be "
              "less than or equal to the maximum batch[%d].",
              num_image, infer_batch_size);
          return {};
        }
      }
    }
    //根据批处理大小，调整内存空间
    adjust_memory(infer_batch_size);

    vector<AffineMatrix> affine_matrixs(num_image);
    cudaStream_t stream_ = (cudaStream_t)stream;
    for (int i = 0; i < num_image; ++i)
      //进行仿射变换都操作，同时在CPU和GPU内存之间进行数据搬运
      preprocess(i, images[i], preprocess_buffers_[i], affine_matrixs[i], stream);

    float *bbox_output_device = bbox_predict_.gpu();
    vector<void *> bindings{input_buffer_.gpu(), bbox_output_device};

    if (has_segment_) {
      bindings = {input_buffer_.gpu(), segment_predict_.gpu(), bbox_output_device};
    }

    if (!trt_->forward(bindings, stream)) {
      INFO("Failed to tensorRT forward.");
      return {};
    }

    for (int ib = 0; ib < num_image; ++ib) {
      float *boxarray_device =
          output_boxarray_.gpu() + ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
      float *affine_matrix_device = (float *)preprocess_buffers_[ib]->gpu();
      float *image_based_bbox_output =
          bbox_output_device + ib * (bbox_head_dims_[1] * bbox_head_dims_[2]);
      checkRuntime(cudaMemsetAsync(boxarray_device, 0, sizeof(int), stream_));
      decode_kernel_invoker(image_based_bbox_output, bbox_head_dims_[1], num_classes_,
                            bbox_head_dims_[2], confidence_threshold_, nms_threshold_,
                            affine_matrix_device, boxarray_device, MAX_IMAGE_BOXES, type_, stream_);
    }
    checkRuntime(cudaMemcpyAsync(output_boxarray_.cpu(), output_boxarray_.gpu(),
                                 output_boxarray_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
    checkRuntime(cudaStreamSynchronize(stream_));

    vector<BoxArray> arrout(num_image);
    int imemory = 0;
    for (int ib = 0; ib < num_image; ++ib) {
      float *parray = output_boxarray_.cpu() + ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
      int count = min(MAX_IMAGE_BOXES, (int)*parray);
      BoxArray &output = arrout[ib];
      output.reserve(count);
      for (int i = 0; i < count; ++i) {
        float *pbox = parray + 1 + i * NUM_BOX_ELEMENT;
        int label = pbox[5];
        int keepflag = pbox[6];
        if (keepflag == 1) {
          Box result_object_box(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
          if (has_segment_) {
            int row_index = pbox[7];
            int mask_dim = segment_head_dims_[1];
            float *mask_weights = bbox_output_device +
                                  (ib * bbox_head_dims_[1] + row_index) * bbox_head_dims_[2] +
                                  num_classes_ + 4;

            float *mask_head_predict = segment_predict_.gpu();
            float left, top, right, bottom;
            float *i2d = affine_matrixs[ib].i2d;
            affine_project(i2d, pbox[0], pbox[1], &left, &top);
            affine_project(i2d, pbox[2], pbox[3], &right, &bottom);

            float box_width = right - left;
            float box_height = bottom - top;

            float scale_to_predict_x = segment_head_dims_[3] / (float)network_input_width_;
            float scale_to_predict_y = segment_head_dims_[2] / (float)network_input_height_;
            int mask_out_width = box_width * scale_to_predict_x + 0.5f;
            int mask_out_height = box_height * scale_to_predict_y + 0.5f;

            if (mask_out_width > 0 && mask_out_height > 0) {
              if (imemory >= (int)box_segment_cache_.size()) {
                box_segment_cache_.push_back(std::make_shared<trt::Memory<unsigned char>>());
              }

              int bytes_of_mask_out = mask_out_width * mask_out_height;
              auto box_segment_output_memory = box_segment_cache_[imemory];
              result_object_box.seg =
                  make_shared<InstanceSegmentMap>(mask_out_width, mask_out_height);

              unsigned char *mask_out_device = box_segment_output_memory->gpu(bytes_of_mask_out);
              unsigned char *mask_out_host = result_object_box.seg->data;
              decode_single_mask(left * scale_to_predict_x, top * scale_to_predict_y, mask_weights,
                                 mask_head_predict + ib * segment_head_dims_[1] *
                                                         segment_head_dims_[2] *
                                                         segment_head_dims_[3],
                                 segment_head_dims_[3], segment_head_dims_[2], mask_out_device,
                                 mask_dim, mask_out_width, mask_out_height, stream_);
              checkRuntime(cudaMemcpyAsync(mask_out_host, mask_out_device,
                                           box_segment_output_memory->gpu_bytes(),
                                           cudaMemcpyDeviceToHost, stream_));
            }
          }
          output.emplace_back(result_object_box);
        }
      }
    }

    if (has_segment_) checkRuntime(cudaStreamSynchronize(stream_));

    return arrout;
  }
};

Infer *loadraw(const std::string &engine_file, Type type, float confidence_threshold,
               float nms_threshold) {
  InferImpl *impl = new InferImpl();
  if (!impl->load(engine_file, type, confidence_threshold, nms_threshold)) {
    delete impl;
    impl = nullptr;
  }
  return impl;
}

shared_ptr<Infer> load(const string &engine_file, Type type, float confidence_threshold,
                       float nms_threshold) {
  return std::shared_ptr<InferImpl>(
      (InferImpl *)loadraw(engine_file, type, confidence_threshold, nms_threshold));
}

std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v) {
  const int h_i = static_cast<int>(h * 6);
  const float f = h * 6 - h_i;
  const float p = v * (1 - s);
  const float q = v * (1 - f * s);
  const float t = v * (1 - (1 - f) * s);
  float r, g, b;
  switch (h_i) {
    case 0:
      r = v, g = t, b = p;
      break;
    case 1:
      r = q, g = v, b = p;
      break;
    case 2:
      r = p, g = v, b = t;
      break;
    case 3:
      r = p, g = q, b = v;
      break;
    case 4:
      r = t, g = p, b = v;
      break;
    case 5:
      r = v, g = p, b = q;
      break;
    default:
      r = 1, g = 1, b = 1;
      break;
  }
  return make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255),
                    static_cast<uint8_t>(r * 255));
}

std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id) {
  float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
  float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
  return hsv2bgr(h_plane, s_plane, 1);
}

};  // namespace yolo