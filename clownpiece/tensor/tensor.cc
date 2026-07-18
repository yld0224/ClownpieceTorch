#include "meta.h"
#include "parallel.h"
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
  const int Tensor::offset_atv(veci v) const {
    int physical_idx = offset_;
    int sz = v.size();
    for (int i = 0; i < sz; ++i) {
      physical_idx += v[i] * stride_[i];
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
  void Tensor::apply_unary_functions(Tensor& tensor, std::function<dtype(dtype)> func) {
    int num = tensor.numel();
    parallel_for(0, num, 16384, [&](int begin, int end) {
      for (int i = begin; i < end; ++i) {
        tensor.data_at(i) = func(tensor.data_at(i));
      }
    });
  }

  Tensor Tensor::apply_binary_functions(const Tensor& t1, const Tensor& t2, std::function<dtype(dtype, dtype)> func) {
    Tensor ret = Tensor(t1.shape_);
    int num = t1.numel();
    parallel_for(0, num, 16384, [&](int begin, int end) {
      for (int i = begin; i < end; ++i) {
        ret.data_at(i) = func(t1.data_at(i), t2.data_at(i));
      }
    });
    return ret;
  }

  Tensor Tensor::operator-() const {
    Tensor ret = clone();
    Tensor::apply_unary_functions(ret, [](dtype data) -> dtype{return -data;});
    return ret;
  }

  Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
    std::pair<Tensor, Tensor> p = Tensor::broadcast(lhs, rhs);
    return Tensor::apply_binary_functions(p.first, p.second, [](dtype a, dtype b)->dtype{return a + b;});
  }
  
  Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
    std::pair<Tensor, Tensor> p = Tensor::broadcast(lhs, rhs);
    return Tensor::apply_binary_functions(p.first, p.second, [](dtype a, dtype b)->dtype{return a - b;});
  }

  Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
    std::pair<Tensor, Tensor> p = Tensor::broadcast(lhs, rhs);
    return Tensor::apply_binary_functions(p.first, p.second, [](dtype a, dtype b)->dtype{return a * b;});
  }

  Tensor operator/(const Tensor& lhs, const Tensor& rhs) {
    std::pair<Tensor, Tensor> p = Tensor::broadcast(lhs, rhs);
    return Tensor::apply_binary_functions(p.first, p.second, [](dtype a, dtype b)->dtype{return a / b;});
  }
  
  Tensor operator==(const Tensor& lhs, const Tensor& rhs) {
    std::pair<Tensor, Tensor> p = Tensor::broadcast(lhs, rhs);
    return Tensor::apply_binary_functions(p.first, p.second, [](dtype a, dtype b)->dtype{return a == b ? 1.0f : 0.0f;});
  }

  Tensor operator!=(const Tensor& lhs, const Tensor& rhs) {
    std::pair<Tensor, Tensor> p = Tensor::broadcast(lhs, rhs);
    return Tensor::apply_binary_functions(p.first, p.second, [](dtype a, dtype b)->dtype{return a != b ? 1.0f : 0.0f;});
  }
  Tensor operator<(const Tensor& lhs, const Tensor& rhs) {
     std::pair<Tensor, Tensor> p = Tensor::broadcast(lhs, rhs);
    return Tensor::apply_binary_functions(p.first, p.second, [](dtype a, dtype b)->dtype{return a < b ? 1.0f : 0.0f;});
  }

  Tensor operator<=(const Tensor& lhs, const Tensor& rhs) {
     std::pair<Tensor, Tensor> p = Tensor::broadcast(lhs, rhs);
    return Tensor::apply_binary_functions(p.first, p.second, [](dtype a, dtype b)->dtype{return a <= b ? 1.0f : 0.0f;});
  }

  Tensor operator>=(const Tensor& lhs, const Tensor& rhs) {
     std::pair<Tensor, Tensor> p = Tensor::broadcast(lhs, rhs);
    return Tensor::apply_binary_functions(p.first, p.second, [](dtype a, dtype b)->dtype{return a >= b ? 1.0f : 0.0f;});
  }

  Tensor operator>(const Tensor& lhs, const Tensor& rhs) {
     std::pair<Tensor, Tensor> p = Tensor::broadcast(lhs, rhs);
    return Tensor::apply_binary_functions(p.first, p.second, [](dtype a, dtype b)->dtype{return a > b ? 1.0f : 0.0f;});
  }

  /*
    matrix multiplication
  */
  Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.dim() == 0 || rhs.dim() == 0) {
      throw std::runtime_error("Matmul: invalid scalar inputs");
    }
    if (lhs.dim() == 1 && rhs.dim() == 1) {
      if (lhs.shape_[0] != rhs.shape_[0]) {throw std::runtime_error("Matmul: invalid shape");}
      int sz = lhs.shape_[0];
      dtype ret = 0;
      for (int i = 0; i < sz; ++i) {
        ret += lhs.get_data_at(i) * rhs.get_data_at(i);
      }
      return Tensor(ret);
    }
    Tensor l = lhs;
    Tensor r = rhs;
    if (lhs.dim() == 1 && rhs.dim() >= 2) {
      l = lhs.unsqueeze(0);
      return matmul(l, r).squeeze(r.dim() - 2);
    }
    if (rhs.dim() == 1 && lhs.dim() >= 2) {
      r = rhs.unsqueeze(1);
      return matmul(l, r).squeeze(l.dim() - 1);
    }
    if (r.shape_[r.dim() - 2] != l.shape_[l.dim() - 1]) {throw std::runtime_error("Matmul: invalid shape");}
    std::pair<Tensor, Tensor> p = Tensor::broadcast_for_matmul(l, r);
    Tensor l1 = p.first;
    Tensor r1 = p.second;
    shape_t ret_shape = r1.shape_;
    int d1 = l1.dim() - 1;
    int d2 = r1.dim() - 2;
    ret_shape[d2] = l.shape_[l.dim() - 2];
    Tensor ret = Tensor(ret_shape);
    int num = ret.numel();
    const int reduction_size = l1.shape_[d1];
    const int grain_size = std::max(1, 32768 / std::max(1, reduction_size));
    parallel_for(0, num, grain_size, [&](int begin, int end) {
      for (int i = begin; i < end; ++i) {
        int tmp = i;
        veci coord(ret_shape.size());
        for (int j = ret_shape.size() - 1; j >= 0; --j) {
          int idx = tmp % ret_shape[j];
          tmp /= ret_shape[j];
          coord[j] = idx;
        }
        veci coord_l1 = coord;
        veci coord_r1 = coord;
        dtype n = 0;
        for (int j = 0; j < reduction_size; ++j) {
          coord_l1[d1] = j;
          coord_r1[d2] = j;
          n += l1.storage_[l1.offset_atv(coord_l1)] * r1.storage_[r1.offset_atv(coord_r1)];
        }
        ret.storage_[ret.offset_atv(coord)] = n;
      }
    });
    return ret;
  }

  std::pair<Tensor, Tensor> Tensor::broadcast_for_matmul(const Tensor& lhs, const Tensor& rhs) {
    int lb = lhs.dim() - 2;
    int rb = rhs.dim() - 2;
    int bd = std::max(lb, rb);
    shape_t batch_shape(bd, 1);
    auto merge_batch = [&](const Tensor& t) {
      int tb = t.dim() - 2;
      int pad = bd - tb;
      for (int i = 0; i < tb; ++i) {
        int axis = pad + i;
        int old_size = t.size(i);
        int cur_size = batch_shape[axis];
        if (cur_size == old_size) continue;
        if (cur_size == 1) batch_shape[axis] = old_size;
        else if (old_size == 1) continue;
        else throw std::runtime_error("Matmul: batch dimensions cannot broadcast");
      }
    };
    merge_batch(lhs);
    merge_batch(rhs);
    shape_t l_shape = batch_shape;
    l_shape.push_back(lhs.size(lhs.dim() - 2));
    l_shape.push_back(lhs.size(lhs.dim() - 1));
    shape_t r_shape = batch_shape;
    r_shape.push_back(rhs.size(rhs.dim() - 2));
    r_shape.push_back(rhs.size(rhs.dim() - 1));
    return {lhs.broadcast_to(l_shape), rhs.broadcast_to(r_shape)};
  }

  Tensor operator^(const Tensor& lhs, const Tensor& rhs) {
    return matmul(lhs, rhs);
  }

  /*
    other mathematical operations
  */
  Tensor Tensor::sign() const {
    Tensor ret = clone();
    Tensor::apply_unary_functions(ret, [](dtype data) -> dtype{
      if (data > 0) {return 1;}
      if (data == 0) {return 0;}
      return -1;
    });
    return ret;
  }

  Tensor Tensor::abs() const {
    Tensor ret = clone();
    Tensor::apply_unary_functions(ret, [](dtype data) -> dtype{
      return data >= 0 ? data : -data;
    });
    return ret;
  }
  Tensor abs(const Tensor& tensor) {
    Tensor ret = tensor.clone();
    Tensor::apply_unary_functions(ret, [](dtype data) -> dtype{
      return data >= 0 ? data : -data;
    });
    return ret;
  }

  Tensor Tensor::sin() const {
    Tensor ret = clone();
    Tensor::apply_unary_functions(ret, [](dtype data) -> dtype{
      return std::sin(data);
    });
    return ret;
  }
  Tensor sin(const Tensor& tensor) {
    Tensor ret = tensor.clone();
    Tensor::apply_unary_functions(ret, [](dtype data) -> dtype{
      return std::sin(data);
    });
    return ret;
  }

  Tensor Tensor::cos() const {
    Tensor ret = clone();
    Tensor::apply_unary_functions(ret, [](dtype data) -> dtype{
      return std::cos(data);
    });
    return ret;
  }
  Tensor cos(const Tensor& tensor) {
     Tensor ret = tensor.clone();
    Tensor::apply_unary_functions(ret, [](dtype data) -> dtype{
      return std::cos(data);
    });
    return ret;
  }
  Tensor Tensor::tanh() const {
     Tensor ret = clone();
    Tensor::apply_unary_functions(ret, [](dtype data) -> dtype{
      return std::tanh(data);
    });
    return ret;
  }
  Tensor tanh(const Tensor& tensor) {
    Tensor ret = tensor.clone();
    Tensor::apply_unary_functions(ret, [](dtype data) -> dtype{
      return std::tanh(data);
    });
    return ret;
  }

  Tensor Tensor::clamp(dtype min, dtype max) const {
    Tensor ret = clone();
    Tensor::apply_unary_functions(ret, [min, max](dtype data) -> dtype{
      return std::min(max, std::max(min, data));
    });
    return ret;
  }

  Tensor clamp(const Tensor& tensor, dtype min, dtype max) {
    Tensor ret = tensor.clone();
    Tensor::apply_unary_functions(ret, [min, max](dtype data) -> dtype{
      return std::min(max, std::max(min, data));
    });
    return ret;
  }

  Tensor Tensor::log() const {
    Tensor ret = clone();
    Tensor::apply_unary_functions(ret, [](dtype data) -> dtype{
      return std::log(data);
    });
    return ret;
  }

  Tensor log(const Tensor& tensor) {
    Tensor ret = tensor.clone();
    Tensor::apply_unary_functions(ret, [](dtype data) -> dtype{
      return std::log(data);
    });
    return ret;
  }

  Tensor Tensor::exp() const {
    Tensor ret = clone();
    Tensor::apply_unary_functions(ret, [](dtype data) -> dtype{
      return std::exp(data);
    });
    return ret;
  }

  Tensor exp(const Tensor& tensor) {
    Tensor ret = tensor.clone();
    Tensor::apply_unary_functions(ret, [](dtype data) -> dtype{
      return std::exp(data);
    });
    return ret;
  }

  Tensor Tensor::pow(dtype exponent) const {
    Tensor ret = clone();
    Tensor::apply_unary_functions(ret, [exponent](dtype data) -> dtype{
      return std::pow(data, exponent);
    });
    return ret;
  }

  Tensor pow(const Tensor& tensor, dtype exponent) {
    Tensor ret = tensor.clone();
    Tensor::apply_unary_functions(ret, [exponent](dtype data) -> dtype{
      return std::pow(data, exponent);
    });
    return ret;
  }

  Tensor Tensor::sqrt() const {
    Tensor ret = clone();
    Tensor::apply_unary_functions(ret, [](dtype data) -> dtype{
      return std::sqrt(data);
    });
    return ret;
  }

  Tensor sqrt(const Tensor& tensor) {
    Tensor ret = tensor.clone();
    Tensor::apply_unary_functions(ret, [](dtype data) -> dtype{
      return std::sqrt(data);
    });
    return ret;
  }

  Tensor Tensor::sum(int dim, bool keepdims) const {
    int d = this->dim();
    if (dim < 0) dim += d;
    if (dim < 0 || dim >= d) {
      throw std::runtime_error("Sum: invalid dimension");
    }
    shape_t out_shape = shape_;
    out_shape[dim] = 1;
    Tensor ret(out_shape, 0.0f);
    for (int i = 0; i < shape_[dim]; ++i) {
      ret = ret + narrow(dim, i, 1, false);
    }
    if (keepdims) {return ret;}
    return ret.squeeze(dim);
  }

  Tensor sum(const Tensor& tensor, int dim, bool keepdims) {
    return tensor.sum(dim, keepdims);
  }

  std::pair<Tensor, Tensor> Tensor::max(int dim, bool keepdims) const {
    int d = this->dim();
    if (dim < 0) dim += d;
    if (shape_[dim] == 0) {throw std::runtime_error("Max: cannot reduce empty dimension");}
    Tensor values = narrow(dim, 0, 1, false).clone();
    Tensor indices(values.get_shape(), 0.0f);
    for (int k = 1; k < shape_[dim]; ++k) {
      Tensor cur = narrow(dim, k, 1, false);
      for (int i = 0; i < values.numel(); ++i) {
        if (cur.data_at(i) > values.data_at(i)) {
          values.data_at(i) = cur.data_at(i);
          indices.data_at(i) = static_cast<dtype>(k);
        }
      }
    }
    if (keepdims) {return {values, indices};}
    return {values.squeeze(dim), indices.squeeze(dim)};
  }

  std::pair<Tensor, Tensor> max(const Tensor& tensor, int dim, bool keepdims) {
    return tensor.max(dim, keepdims);
  }

  Tensor Tensor::softmax(int dim) const {
    Tensor e = exp();
    return e / e.sum(dim, true);
  }
  Tensor softmax(const Tensor& tensor, int dim) {
    return tensor.softmax(dim);
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
    int d = this->dim();
    if (dim < 0) {dim = d + dim;}
    if (dim < 0 || dim >= d) {throw std::runtime_error("Narrow: invalid dimension");}
    if (length < 0 || start < 0 || 
      start > shape_[dim] || start + length > shape_[dim]) {throw std::runtime_error("Narrow: invalid slicing");}
    vec<slice_t> slices(d);
    for (int i = 0; i < d; ++i) {
      if (i == dim) {
        slices[i] = {start, start + length};
        continue;
      }
      slices[i] = {0, shape_[i]};
    }
    Tensor tmp = (*this)[slices];
    if (copy) {return tmp.clone();}
    return tmp;
  }

  vec<Tensor> Tensor::chunk(int chunks, int dim) const {
    int d = this->dim();
    if (dim < 0) dim += d;
    if (dim < 0 || dim >= d) throw std::runtime_error("Chunk: invalid dimension");
    if (chunks <= 0) throw std::runtime_error("Chunk: chunks must be positive");
    int n = shape_[dim];
    int chunk_size = (n + chunks - 1) / chunks;
    vec<Tensor> ret;
    for (int start = 0; start < n; start += chunk_size) {
      int len = std::min(chunk_size, n - start);
      ret.push_back(narrow(dim, start, len, false));
    }
    return ret;
  }

  vec<Tensor> Tensor::split(int dim, int split_size) const {
    int d = this->dim();
    if (dim < 0) {dim = d + dim;}
    if (dim < 0 || dim >= d) {throw std::runtime_error("Split: invalid dimension");}
    if (split_size <= 0) {throw std::runtime_error("Split: invalid split");}
    vec<Tensor> ret;
    for (int i = 0; i < shape_[dim]; i += split_size) {
      int len = std::min(split_size, shape_[dim] - i);
      ret.push_back(narrow(dim, i, len, false));
    }
    return ret;
  }
  vec<Tensor> Tensor::split(int dim, veci split_sections) const {
    int d = this->dim();
    int sz = split_sections.size();
    if (dim < 0) {dim = d + dim;}
    if (dim < 0 || dim >= d) {throw std::runtime_error("Split: invalid dimension");}
    veci prefix(sz + 1, 0);
    for (int i = 1; i < sz + 1; ++i) {
      prefix[i] = split_sections[i - 1] + prefix[i - 1];
    }
    if (prefix.back() != shape_[dim]) {
      throw std::runtime_error("Split: invalid split");
    }
    vec<Tensor> ret;
    for (int i = 0; i < sz; ++i) {
      ret.push_back(narrow(dim, prefix[i], split_sections[i], false));
    }
    return ret;
  }

  Tensor Tensor::stack(const vec<Tensor>& inputs, int dim) {
    if (inputs.empty()) {throw std::runtime_error("Stack: no input");}
    Tensor t = inputs[0];
    veci ori_shape = t.get_shape();
    for (auto v : inputs) {
      if (v.dim() != t.dim()) {throw std::runtime_error("Stack: dimension unmatch");}
      if (v.get_shape() != ori_shape) {throw std::runtime_error("Stack: shape unmatch");}
    }
    if (dim < 0) {dim = t.dim() + dim + 1;}
    if (dim < 0 || dim > t.dim()) {throw std::runtime_error("Stack: invalid dimension");}
    veci shape(t.dim() + 1);
    for (int i = 0; i < dim; ++i) {
      shape[i] = ori_shape[i];
    }
    shape[dim] = inputs.size();
    for (int i = dim + 1; i < shape.size(); ++i) {
      shape[i] = ori_shape[i - 1];
    }
    int num = t.numel();
    int d = t.dim();
    Tensor ret = Tensor(shape);
    for(int i = 0; i < inputs.size(); ++i) {
      auto& v = inputs[i];
      for (int j = 0; j < num; ++j) {
        int offset = i * ret.stride_[dim];;
        int tmp = j;
        for (int k = d - 1; k >= 0; --k) {
          int idx = tmp % v.shape_[k];
          tmp /= v.shape_[k];
          offset += (k >= dim ? idx * ret.stride_[k + 1] : idx * ret.stride_[k]);
        }
        ret.storage_[offset] = v.storage_[v.offset_at(j)];
      }
    }
    return ret;
  }

  Tensor Tensor::cat(const vec<Tensor>& inputs, int dim) {
    if (inputs.empty()) {throw std::runtime_error("Cat: no input");}
    Tensor t = inputs[0];
    veci ori_shape = t.get_shape();
    int sum = 0;
    if (dim < 0) {dim = t.dim() + dim;}
    if (dim < 0 || dim >= t.dim()) {throw std::runtime_error("Cat: invalid dimension");}
    for (auto v : inputs) {
      if (v.dim() != t.dim()) {throw std::runtime_error("Cat: dimension unmatch");}
      veci s1 = v.get_shape();
      for (int i = 0; i < ori_shape.size(); ++i) {
        if (s1[i] != ori_shape[i] && i != dim) {throw std::runtime_error("Cat: dimension unmatch");}
        if (i == dim) {sum += s1[i];}
      }
    }
    veci shape(ori_shape);
    shape[dim] = sum;
    int d = t.dim();
    Tensor ret = Tensor(shape);
    int dim_offset = 0;
    for(int i = 0; i < inputs.size(); ++i) {
      auto& v = inputs[i];
      int num = v.numel();
      for (int j = 0; j < num; ++j) {
        int offset = 0;
        int tmp = j;
        for (int k = d - 1; k >= 0; --k) {
          int idx = tmp % v.shape_[k];
          tmp /= v.shape_[k];
          offset += (k != dim ? idx * ret.stride_[k] : (idx + dim_offset) * ret.stride_[k]);
        }
        ret.storage_[offset] = v.storage_[v.offset_at(j)];
      }
      dim_offset += v.shape_[dim];
    }
    return ret;
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
    if (dim < 0) {dim += d + 1;}
    if (dim < 0 || dim > d){throw std::runtime_error("Unsqueeze: invalid dimension");}
    shape_t shape(d + 1);
    stride_t stride(d + 1);
    for (int i = 0; i < dim; ++i) {
      shape[i] = shape_[i];
      stride[i] = stride_[i];
    }
    shape[dim] = 1;
    stride[dim] = (dim == d ? 1 : stride_[dim] * shape_[dim]);
    for (int i = dim; i < d; ++i) {
      shape[i + 1] = shape_[i];
      stride[i + 1] = stride_[i];
    }
    return Tensor(shape, stride, offset_, storage_);
  }

  Tensor Tensor::broadcast_to(const shape_t& shape) const {
    if (shape_.empty() && numel_ == 0) {
      throw std::runtime_error("Broadcast_to: tensor has no data");
    }
    veci ret_shape(shape.size(), 1);
    veci ret_stride(shape.size(), 0);
    if (shape_.size() > shape.size()) {throw std::runtime_error("Broadcast_to: invalid shape");}
    int s = std::max(static_cast<int>(shape.size() - shape_.size()), 0);
    int sz = ret_shape.size();
    for (int i = s; i < sz; ++i) {
      ret_shape[i] = shape_[i - s];
      ret_stride[i] = stride_[i - s];
    }
    for (int i = sz - 1; i >= 0; --i) {
      if (ret_shape[i] == shape[i]) {continue;}
      if (ret_shape[i] > shape[i]) {throw std::runtime_error("Broadcast_to: invalid shape");}
      if (ret_shape[i] != 1) {throw std::runtime_error("Broadcast_to: invalid shape");}
      ret_shape[i] = shape[i];
      ret_stride[i] = 0;
    }
    return Tensor(ret_shape, ret_stride, offset_, storage_);
  }

  std::pair<Tensor, Tensor> Tensor::broadcast(const Tensor& lhs, const Tensor& rhs) {
    int sz = std::max(lhs.shape_.size(),rhs.shape_.size());
    if (lhs.shape_.empty() && lhs.numel_ == 0) {
      throw std::runtime_error("Broadcast_to: tensor has no data");
    }
    if (rhs.shape_.empty() && rhs.numel_ == 0) {
      throw std::runtime_error("Broadcast_to: tensor has no data");
    }
    veci l_shape(sz, 1);
    veci l_stride(sz, 0);
    veci r_shape(sz, 1);
    veci r_stride(sz, 0);
    for (int i = (sz - lhs.shape_.size()); i < sz; ++i) {
      l_shape[i] = lhs.shape_[i - (sz - lhs.shape_.size())];
      l_stride[i] = lhs.stride_[i - (sz - lhs.shape_.size())];
    }
    for (int i = (sz - rhs.shape_.size()); i < sz; ++i) {
      r_shape[i] = rhs.shape_[i - (sz - rhs.shape_.size())];
      r_stride[i] = rhs.stride_[i - (sz - rhs.shape_.size())];
    }
    for (int i = sz - 1; i >= 0; --i) {
      if (l_shape[i] == r_shape[i]) {continue;}
      if (l_shape[i] != r_shape[i]) {
        if (l_shape[i] == 1 && l_shape[i] < r_shape[i]) {
          l_shape[i] = r_shape[i];
          l_stride[i] = 0;
        } else if (r_shape[i] == 1 && r_shape[i] < l_shape[i]) {
          r_shape[i] = l_shape[i];
          r_stride[i] = 0;
        } else {
          throw std::runtime_error("Broadcast: invalid shape");
        }
      }
    }
    return {Tensor(l_shape, l_stride, lhs.offset_, lhs.storage_),
       Tensor(r_shape, r_stride, rhs.offset_, rhs.storage_)};
  }
  vec<Tensor> Tensor::broadcast(const vec<Tensor>& tensors) {
    if (tensors.empty()) {
      throw std::runtime_error("Broadcast: no input");
    }
    int max_dim = 0;
    for (const auto& t : tensors) {
      if (t.shape_.empty() && t.numel_ == 0) {
        throw std::runtime_error("Broadcast: tensor has no data");
      }
      max_dim = std::max(max_dim, t.dim());
    }
    shape_t common_shape(max_dim, 1);
    for (const auto& t : tensors) {
      int pad = max_dim - t.dim();
      for (int i = 0; i < t.dim(); ++i) {
        int axis = pad + i;
        int old_size = t.shape_[i];
        int cur_size = common_shape[axis];
        if (cur_size == old_size) {
          continue;
        }
        if (cur_size == 1) {
          common_shape[axis] = old_size;
        } else if (old_size == 1) {
          continue;
        } else {
          throw std::runtime_error("Broadcast: invalid shape");
        }
      }
    }
    vec<Tensor> ret;
    ret.reserve(tensors.size());
    for (const auto& t : tensors) {
      ret.push_back(t.broadcast_to(common_shape));
    }
    return ret;
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
    int d = this->dim();
    if (dim < 0) {dim += d;}
    if (dim < 0 || dim >= d){throw std::runtime_error("Mean: invalid dimension");}
    return sum(dim, keepdims) / shape_[dim];
  }

  Tensor Tensor::var(int dim, bool keepdims, bool unbiased) const {
    int d = this->dim();
    if (dim < 0) {dim += d;}
    if (dim < 0 || dim >= d) {throw std::runtime_error("Var: invalid dimension");}
    Tensor avg = mean(dim, true);
    shape_t out_shape = shape_;
    out_shape[dim] = 1;
    Tensor t(out_shape, 0.0f);
    for (int i = 0; i < shape_[dim]; ++i) {
      Tensor s = narrow(dim, i, 1, false);
      t = t + (s - avg) * (s - avg);
    }
    int n = unbiased ? shape_[dim] - 1 : shape_[dim];
    if (keepdims) {return t / n;}
    return (t / n).squeeze(dim);
  }
};
