using Numnet.Tensor.Base;

namespace Numnet.Tensor.Common{
    public interface ITensor<T> where T: struct{
        public Span<T> AsSpan();
        public TensorMemory<T> TMemory{ get; }
    }
    public interface ITensor{
        internal TensorLayout TLayout{ get; }
    }
}