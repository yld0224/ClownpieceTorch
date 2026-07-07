#include "meta.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#include "tensor.h"
#include <cmath>
#include <string>



namespace at {

  /*
    utils for printing
  */
  // print vector int
  std::ostream& operator<<(std::ostream& os, const shape_t& shape) {
    os << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
      os << shape[i];
      if (i < shape.size() - 1)
        os << ", ";
    }
    os << ")";
    return os;
  }

  int print_tensor_data_recursive(std::ostream& os, const Tensor& tensor, int dim_index, int data_index, std::string prefix) {
    if (tensor.dim() == 0) {
      if (tensor.numel() == 0)
        os << "[]";
      else
        os << tensor.data_at(0);
      return 0;
    }
    os << "[";
    if (dim_index == tensor.dim() - 1 || tensor.dim() == 0) {
      for (int i = 0; i < tensor.size(dim_index); ++i) {
        os << tensor.data_at(data_index++);
        if (i < tensor.size(dim_index) - 1)
            os << ", ";
      }
    } else {

      for (int i = 0; i < tensor.size(dim_index); ++i) {
        if (i > 0)
          os << "\n" << prefix;
        data_index = print_tensor_data_recursive(os, tensor, dim_index + 1, data_index, prefix + " ");
        if (i < tensor.size(dim_index) - 1)
          os << ",";
      }
    }
    os << "]";
    return data_index;
  }

  std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << "Tensor(\n  shape=" << tensor.shape_ << ", strides=" << tensor.stride_ << "\n  data={\n";
    std::string prefix = "    ";
    os << prefix;
    print_tensor_data_recursive(os, tensor, 0, 0, prefix + " ");
    os << "\n  }\n)\n";
    return os;
  }

  /*
    Begin your implement here !
  */

  dtype& Tensor::data_at(int index) const {
    if (index < 0 || index >= numel_) {throw std::runtime_error("Data_at: invalid index");}
    return storage_[offset_at(index)]; 
  }

  /*
    constructors and assignments
  */
  Tensor::Tensor() : shape_(shape_t{}), stride_(stride_t{}), offset_(0), storage_(),
    numel_(0), is_contiguous_(true) {}
  Tensor::Tensor(dtype value) : shape_(shape_t{}), stride_(stride_t{}), offset_(0), storage_(1, value),
    numel_(1), is_contiguous_(true) {}
  Tensor::Tensor(const shape_t& shape) : shape_(shape), offset_(0), is_contiguous_(true) {
    int size = shape.size();
    stride_.resize(size);
    if (size == 0) {
      storage_ = Storage(1);
      numel_ = 1;
      return;
    }
    stride_[size - 1] = 1;
    for (int i = size - 2; i >= 0; --i) {
      stride_[i] = stride_[i + 1] * shape_[i + 1];
    }
    numel_ = stride_[0] * shape_[0];
    storage_ = Storage(numel_);
  }
  Tensor::Tensor(const shape_t& shape, dtype value) : shape_(shape), offset_(0), is_contiguous_(true) {
    int size = shape.size();
    stride_.resize(size);
    if (size == 0) {
      storage_ = Storage(1, value);
      numel_ = 1;
      return;
    }
    stride_[size - 1] = 1;
    for (int i = size - 2; i >= 0; --i) {
      stride_[i] = stride_[i + 1] * shape_[i + 1];
    }
    numel_ = stride_[0] * shape_[0];
    storage_ = Storage(numel_, value);
  }
  Tensor::Tensor(const shape_t& shape, std::function<dtype()> generator) : shape_(shape), offset_(0), is_contiguous_(true) {
    int size = shape.size();
    stride_.resize(size);
    if (size == 0) {
      storage_ = Storage(1, generator);
      numel_ = 1;
      return;
    }
    stride_[size - 1] = 1;
    for (int i = size - 2; i >= 0; --i) {
      stride_[i] = stride_[i + 1] * shape_[i + 1];
    }
    numel_ = stride_[0] * shape_[0];
    storage_ = Storage(numel_, generator);
  }
  Tensor::Tensor(const shape_t& shape, const vec<dtype>& data) : shape_(shape), offset_(0), storage_(data), is_contiguous_(true) {
    int size = shape.size();
    if (size == 0) {
      stride_ = stride_t{};
      numel_ = 1;
      return;
    }
    stride_.resize(size);
    stride_[size - 1] = 1;
    for (int i = size - 2; i >= 0; --i) {
      stride_[i] = stride_[i + 1] * shape_[i + 1];
    }
    numel_ = stride_[0] * shape_[0];
  }
  Tensor::Tensor(const shape_t& shape, const stride_t& stride, int offset, Storage storage) : shape_(shape), stride_(stride), offset_(offset), storage_(storage) {
    if (shape.empty()) {
      numel_ = 1;
      is_contiguous_ = true;
      return;
    }
    bool f = true;
    int size = shape.size();
    int n = shape[size - 1];
    if (stride_[size - 1] != 1) {f = false;}
    for (int i = size - 2; i >= 0; --i) {
      if (stride_[i] != stride_[i + 1] * shape[i + 1]) {
        f = false;
      }
      n *= shape[i];
    }
    numel_ = n;
    is_contiguous_ = f;
  }

  Tensor::Tensor(const Tensor& other) = default;

  Tensor& Tensor::operator=(const Tensor& other) = default;

  Tensor& Tensor::operator=(dtype value) {
    if (numel() != 1) {
      throw std::runtime_error("Scalar: only valid for singleton tensor");
    }
    storage_[offset_] = value;
    return *this;
  }

  /* 
    destructor
  */
  Tensor::~Tensor() = default;


  /*
    convert to dtype value
    only valid for singleton tensor
  */
  dtype Tensor::item() const {
     if (numel() != 1) {
      throw std::runtime_error("Item: only valid for singleton tensor");
    }
    return storage_[offset_];
  }

  /*
    utils
  */

  int Tensor::numel() const {
    return numel_;
  }

  int Tensor::dim() const {
    return shape_.size();
  }

  veci Tensor::size() const {
    return shape_;
  }

  int Tensor::size(int dim) const {
    int n = shape_.size();
    if (dim >= n || dim < -n) {
      throw std::runtime_error("Size: invalid dimension");
    }
    if (dim >= 0) {return shape_[dim];}
    else {return shape_[dim + n];}
  }

  bool Tensor::is_contiguous() const {
    return is_contiguous_;
  }

  const int Tensor::offset_at(int n) const {
    int physical_idx = offset_;
    int dim = shape_.size();
    for (int j = dim - 1; j >= 0; --j) {
      int idx = n % shape_[j];
      n /= shape_[j];
      physical_idx += idx * stride_[j];
    }
    return physical_idx;
  }


  /*
    clone, make contiguous, copy_ and scatter
  */
  Tensor Tensor::clone() const {
    if (numel_ == 0) {
      if (!shape_.empty()) {return Tensor(shape_);}
      return Tensor();
    }
    Tensor ret = Tensor(shape_);
    ret.is_contiguous_ = true;
    int dim = shape_.size();
    for (int i = 0; i < numel_; ++i) {
      ret.storage_[i] = storage_[offset_at(i)];
    }
    return ret;
  }

  Tensor Tensor::contiguous() const {
    if (is_contiguous_) {
      return Tensor(*this);
    }
    return this->clone();
  }

  Tensor Tensor::copy_(const Tensor& other) const {
    if (shape_ != other.shape_) {throw std::runtime_error("Copy: shape unmatch");}
    for (int i = 0; i < other.numel_; ++i) {
      storage_[offset_at(i)] = other.storage_[other.offset_at(i)];
    }
    return *this;
  }

  Tensor Tensor::scatter_(int dim, const Tensor& index, const Tensor& src) const {
    int d = src.dim();
    if (dim < 0) {dim = d + dim;}
    if (dim < 0 || dim >= d) {throw std::runtime_error("Scatter: dimension out of bound");}
    if (index.dim() != d || this->dim() != d) {
      throw std::runtime_error("Scatter: dimension unmatch");
    }
    for (int i = 0; i < d; ++i) {
      if (index.shape_[i] != src.shape_[i]) {
        throw std::runtime_error("Scatter: dimension unmatch between src and index");
      }
      if (i != dim && shape_[i] != src.shape_[i]){
        throw std::runtime_error("Scatter: dimension unmatch between src and this tensor");
      }
    }
    for (int i = 0; i < src.numel_; ++i) {
      dtype index_data = index.storage_[index.offset_at(i)];
      int ret_idx = offset_;
      int src_idx = src.offset_;
      int tmp = i;
      for (int j = d - 1; j >= 0; --j) {
        int idx = tmp % src.shape_[j];
        tmp /= src.shape_[j];
        src_idx += idx * src.stride_[j];
        ret_idx += (j == dim ? index_data * stride_[j] : idx * stride_[j]);
      }
      storage_[ret_idx] = src.storage_[src_idx];
    }
    return *this;
  }


  /*
    subscriptor
  */

  Tensor Tensor::operator[](const vec<slice_t>& slices) const {
    if (slices.size() > dim()) {throw std::runtime_error("Operator[]: too many index dimensions");}
    int offset = offset_;
    veci shape = shape_;
    for (int i = 0; i < slices.size(); ++i) {
      auto [s, t] = slices[i];
      int len = shape_[i];
      if (s < 0) {s = len + s;}
      if (t < 0) {t = len + t;}
      s = std::max(0, std::min(s, len));
      t = std::max(0, std::min(t, len));
      offset += s * stride_[i];
      shape[i] = std::max(0, t - s);
    }
    return Tensor(shape, stride_, offset, storage_);
  }

  Tensor Tensor::operator[](slice_t slice) const {
    int offset = offset_;
    veci shape = shape_;
    int len = shape_[0];
    auto [s, t] = slice;
    if (s < 0) {s = len + s;}
    if (t < 0) {t = len + t;}
    s = std::max(0, std::min(s, len));
    t = std::max(0, std::min(t, len));
    offset += s * stride_[0];
    shape[0] = std::max(0, t - s);
    return Tensor(shape, stride_, offset, storage_);
  }

  Tensor Tensor::operator[](const veci& index) const {
    int d = this->dim();
    int sz = index.size();
    if (sz > d) {throw std::runtime_error("Operator[]: too many index dimensions");}
    veci idx = index;
    for (int i = 0; i < sz; ++i) {
      int len = shape_[i];
      if (idx[i] < 0) {idx[i] = len + idx[i];}
      if (idx[i] >= len || idx[i] < 0) {throw std::runtime_error("Operator[]: index out of bound");}
    }
    int offset = offset_;
    for (int i = 0; i < sz; ++i) {
      offset += idx[i] * stride_[i];
    } 
    veci shape(d - sz);
    veci stride(d - sz);
    for (int i = 0; i < d - sz; ++i) {
      shape[i] = shape_[i + sz];
      stride[i] = stride_[i + sz];
    } 
    return Tensor(shape, stride, offset, storage_);
  }

  Tensor Tensor::operator[](int index) const {
    int d = this->dim();
    if (d == 0) {
      throw std::runtime_error("Operator[]: too many indices");
    }
    int offset = offset_;
    int len = shape_[0];
    if (index < 0) {index = len + index;}
    if (index >= len || index < 0) {throw std::runtime_error("Operator[]: index out of bound");}
    offset += index * stride_[0];
    veci shape(d - 1);
    veci stride(d - 1);
    for (int i = 0; i < d - 1; ++i) {
      shape[i] = shape_[i + 1];
      stride[i] = stride_[i + 1];
    } 
    return Tensor(shape, stride, offset, storage_);
  }

  /*
    operators
  */
  void Tensor::apply_unary_functions(Tensor& tensor, std::function<dtype(dtype)> func) const {
    int num = tensor.numel();
    for (int i = 0; i < num; ++i) {
      tensor.data_at(i) = func(tensor.data_at(i));
    }
  }

  Tensor Tensor::operator-() const {
    Tensor ret = clone();
    apply_unary_functions(ret, [](dtype data) -> dtype{return -data;});
    return ret;
  }

  Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
    throw std::runtime_error("Unimplemented");
  }
  
  Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
    throw std::runtime_error("Unimplemented");
  }

  Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
    throw std::runtime_error("Unimplemented");
  }

  Tensor operator/(const Tensor& lhs, const Tensor& rhs) {
    throw std::runtime_error("Unimplemented");
  }
  
  Tensor operator==(const Tensor& lhs, const Tensor& rhs) {
    throw std::runtime_error("Unimplemented");
  }

  Tensor operator!=(const Tensor& lhs, const Tensor& rhs) {
    throw std::runtime_error("Unimplemented");
  }
  Tensor operator<(const Tensor& lhs, const Tensor& rhs) {
    throw std::runtime_error("Unimplemented");
  }

  Tensor operator<=(const Tensor& lhs, const Tensor& rhs) {
    throw std::runtime_error("Unimplemented");
  }

  Tensor operator>=(const Tensor& lhs, const Tensor& rhs) {
    throw std::runtime_error("Unimplemented");
  }

  Tensor operator>(const Tensor& lhs, const Tensor& rhs) {
    throw std::runtime_error("Unimplemented");
  }

  /*
    matrix multiplication
  */
  Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
    throw std::runtime_error("Unimplemented");
  }

  Tensor operator^(const Tensor& lhs, const Tensor& rhs) {
    throw std::runtime_error("Unimplemented");
  }

  /*
    other mathematical operations
  */
  Tensor Tensor::sign() const {
    throw std::runtime_error("Unimplemented");
  }

  Tensor Tensor::abs() const {
    throw std::runtime_error("Unimplemented");
  }
  Tensor abs(const Tensor& tensor) {
    throw std::runtime_error("Unimplemented");
  }

  Tensor Tensor::sin() const {
    throw std::runtime_error("Unimplemented");
  }
  Tensor sin(const Tensor& tensor) {
    throw std::runtime_error("Unimplemented");
  }

  Tensor Tensor::cos() const {
    throw std::runtime_error("Unimplemented");
  }
  Tensor cos(const Tensor& tensor) {
    throw std::runtime_error("Unimplemented");
  }
  Tensor Tensor::tanh() const {
    throw std::runtime_error("Unimplemented");
  }
  Tensor tanh(const Tensor& tensor) {
    throw std::runtime_error("Unimplemented");
  }

  Tensor Tensor::clamp(dtype min, dtype max) const {
    throw std::runtime_error("Unimplemented");
  }

  Tensor clamp(const Tensor& tensor, dtype min, dtype max) {
    throw std::runtime_error("Unimplemented");
  }

  Tensor Tensor::log() const {
    throw std::runtime_error("Unimplemented");
  }

  Tensor log(const Tensor& tensor) {
    throw std::runtime_error("Unimplemented");
  }

  Tensor Tensor::exp() const {
    throw std::runtime_error("Unimplemented");
  }

  Tensor exp(const Tensor& tensor) {
    throw std::runtime_error("Unimplemented");
  }

  Tensor Tensor::pow(dtype exponent) const {
    throw std::runtime_error("Unimplemented");
  }

  Tensor pow(const Tensor& tensor, dtype exponent) {
    throw std::runtime_error("Unimplemented");
  }

  Tensor Tensor::sqrt() const {
    throw std::runtime_error("Unimplemented");
  }

  Tensor sqrt(const Tensor& tensor) {
    throw std::runtime_error("Unimplemented");
  }

  Tensor Tensor::sum(int dim, bool keepdims) const {
    throw std::runtime_error("Unimplemented");
  }

  Tensor sum(const Tensor& tensor, int dim, bool keepdims) {
    throw std::runtime_error("Unimplemented");
  }

  std::pair<Tensor, Tensor> Tensor::max(int dim, bool keepdims) const {
    throw std::runtime_error("Unimplemented");
  }

  std::pair<Tensor, Tensor> max(const Tensor& tensor, int dim, bool keepdims) {
    throw std::runtime_error("Unimplemented");
  }

  Tensor Tensor::softmax(int dim) const {
    throw std::runtime_error("Unimplemented");
  }
  Tensor softmax(const Tensor& tensor, int dim) {
    throw std::runtime_error("Unimplemented");
  }

  /*
    helper constructor
  */

  Tensor Tensor::ones_like() const {
    return Tensor(this->shape_, 1.0f);
  }
  Tensor Tensor::zeros_like() const {
    return Tensor(this->shape_, 0.0f);
  }
  Tensor Tensor::randn_like() const {
    return Tensor(this->shape_, []()->dtype{return random::randn(random::mt19937_rng, 0, 1);});
  }
  Tensor Tensor::empty_like() const {
   return Tensor(this->shape_);
  }

  /*
    shape manipulation
  */

  Tensor Tensor::permute(veci p) const {
    veci shape = shape_;
    veci stride = stride_;
    int sz = p.size();
    for (int i = 0; i < sz; ++i) {
      if (p[i] < 0){p[i] = dim() + p[i];}
      if (p[i] < 0 || p[i] >= dim()) {throw std::runtime_error("Permute: invalid permutation");}
      shape[i] = shape_[p[i]];
      stride[i] = stride_[p[i]];
    }
    return Tensor(shape, stride, offset_, storage_);
  }

  Tensor Tensor::transpose(int dim1, int dim2) const {
    veci shape = shape_;
    veci stride = stride_;
    int d = dim();
    if (dim1 < 0) {dim1 = d + dim1;}
    if (dim2 < 0) {dim2 = d + dim2;}
    if (dim1 < 0 || dim1 >= d || dim2 < 0 || dim2 >= d) {throw std::runtime_error("Transpose: invalid dimension");}
    shape[dim1] = shape_[dim2];
    shape[dim2] = shape_[dim1];
    stride[dim1] = stride_[dim2];
    stride[dim2] = stride_[dim1];
    return Tensor(shape, stride, offset_, storage_);
  }

  Tensor Tensor::reshape(const shape_t& purposed_shape, bool copy) const {
    int i = -1;
    int size = 1;
    veci stride;
    for (int j = 0; j < purposed_shape.size(); ++j) {
      if (purposed_shape[j] == -1 && i == -1) {
        i = j; 
      } else if (purposed_shape[j] == -1 && i != -1) {
        throw std::runtime_error("Reshape: purposed_shape not match with shape");
      } else {
        size *= purposed_shape[j];
      }
    }
    veci shape = purposed_shape;
    if (i == -1){
      if (numel_ != size) {throw std::runtime_error("Reshape: purposed_shape not match with shape");}
    } else {
      if (numel_ % size != 0) {throw std::runtime_error("Reshape: purposed_shape not match with shape");}
      shape[i] = numel_ / size;
    }
    stride.resize(shape.size());
    if (!stride.empty()){
      stride[stride.size() - 1] = 1;
      for (int j = stride.size() - 2; j >= 0; --j) {
        stride[j] = stride[j + 1] * shape[j + 1];
      }
    }
    if (!copy && is_contiguous_) {
      return Tensor(shape, stride, offset_, storage_);
    }
    Tensor ret = Tensor(shape);
    for (int i = 0; i < numel_; ++i) {
      ret.storage_[i] = storage_[offset_at(i)];
    }
    return ret;
  }

  Tensor Tensor::view(const shape_t& purposed_shape) const {
    if (!is_contiguous_) {throw std::runtime_error("View: tensor not contiguous");}
    return reshape(purposed_shape, false);
  }

  Tensor Tensor::narrow(int dim, int start, int length, bool copy) const {
    throw std::runtime_error("Unimplemented");
  }

  vec<Tensor> Tensor::chunk(int chunks, int dim) const {
    throw std::runtime_error("Unimplemented");
  }

  vec<Tensor> Tensor::split(int dim, int split_size) const {
    throw std::runtime_error("Unimplemented");
  }
  vec<Tensor> Tensor::split(int dim, veci split_sections) const {
    throw std::runtime_error("Unimplemented");
  }

  Tensor Tensor::stack(const vec<Tensor>& inputs, int dim) {
    throw std::runtime_error("Unimplemented");
  }

  Tensor Tensor::cat(const vec<Tensor>& inputs, int dim) {
    throw std::runtime_error("Unimplemented");
  }

  Tensor Tensor::squeeze(int dim) const {
    int d = this->dim();
    if (dim < 0) {dim += d;}
    if (dim < 0 || dim >= d || shape_[dim] != 1){throw std::runtime_error("Squeeze: invalid dimension");}
    shape_t shape(d - 1);
    stride_t stride(d - 1);
    int j = 0;
    for (int i = 0; i < d; ++i) {
      if (i == dim) {continue;}
      shape[j] = shape_[i];
      stride[j] = stride_[i];
      ++j;
    }
    return Tensor(shape, stride, offset_, storage_); 
  }

  Tensor Tensor::unsqueeze(int dim) const {
    int d = this->dim();
    if (dim < 0) {dim += d;}
    if (dim < 0 || dim > d){throw std::runtime_error("Unsqueeze: invalid dimension");}
    shape_t shape(d + 1);
    stride_t stride(d + 1);
    for (int i = 0; i < dim; ++i) {
      shape[i] = shape_[i];
      stride[i] = stride_[i];
    }
    shape[dim] = stride[dim] = 1;
    for (int i = dim; i < d; ++i) {
      shape[i + 1] = shape_[i];
      stride[i + 1] = stride_[i];
    }
    return Tensor(shape, stride, offset_, storage_);
  }

  Tensor Tensor::broadcast_to(const shape_t& shape) const {
    throw std::runtime_error("Unimplemented");
  }

  std::pair<Tensor, Tensor> Tensor::broadcast(const Tensor& lhs, const Tensor& rhs) {
    throw std::runtime_error("Unimplemented");
  }
  vec<Tensor> Tensor::broadcast(const vec<Tensor>& tensors) {
    throw std::runtime_error("Unimplemented");
  }



  /*
    helper constructors
  */
  Tensor to_singleton_tensor(dtype value, int dim) {
    return Tensor(shape_t(dim, 1), value);
  }

  Tensor ones(const shape_t& shape) {
    return Tensor(shape, 1.0f);
  }
  Tensor ones_like(const Tensor& ref) {
    return Tensor(ref.get_shape(), 1.0f);
  }

  Tensor zeros(const shape_t& shape) {
    return Tensor(shape, 0.0f);
  }
  Tensor zeros_like(const Tensor& ref) {
    return Tensor(ref.get_shape(), 0.0f);
  }

  Tensor randn(const shape_t& shape) {
    return Tensor(shape, []()->dtype{return random::randn(random::mt19937_rng, 0, 1);});
  }
  Tensor randn_like(const Tensor& ref) {
    return Tensor(ref.get_shape(), []()->dtype{return random::randn(random::mt19937_rng, 0, 1);});
  }

  Tensor empty(const shape_t& shape) {
    return Tensor(shape);
  }
  Tensor empty_like(const Tensor& ref) {
    return Tensor(ref.get_shape());
  }

  Tensor arange(dtype start, dtype end, dtype step) {
    if (step <= 0) {
      throw std::runtime_error("Arange: step must be positive");
    }
    vec<dtype> data;
    for (dtype x = start; x < end; x += step) {
      data.push_back(x);
    }
    return Tensor(shape_t{static_cast<int>(data.size())}, data);
  }

  Tensor range(dtype start, dtype end, dtype step) {
    if (step <= 0) {
      throw std::runtime_error("Range: step must be positive");
    }
    vec<dtype> data;
    for (dtype x = start; x <= end; x += step) {
      data.push_back(x);
    }
    return Tensor(shape_t{static_cast<int>(data.size())}, data);
  }

  Tensor linspace(dtype start, dtype end, int num_steps) {
    vec<dtype> data(num_steps);
    if (num_steps == 0) {
      return Tensor(shape_t{0}, data);
    }
    if (num_steps == 1) {
      data[0] = start;
      return Tensor(shape_t{1}, data);
    }
    dtype step = (end - start) / static_cast<dtype>(num_steps - 1);
    for (int i = 0; i < num_steps; ++i) {
        data[i] = start + i * step;
    }
    data[num_steps - 1] = end;
    return Tensor(shape_t{num_steps}, data);
  }
  
  /*
    Week3 adds-on
  */
  Tensor Tensor::mean(int dim, bool keepdims) const {
    throw std::runtime_error("Unimplemented");
  }

  Tensor Tensor::var(int dim, bool keepdims, bool unbiased) const {
    throw std::runtime_error("Unimplemented");
  }
};
