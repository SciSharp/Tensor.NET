using Numnet.Native;

namespace Numnet.Tensor.Base{
    public sealed class TensorLayout
    {
        static readonly int MAX_NDIM = 4;
        public DType _dtype { get; private set; }
        public int _ndim{ get; private set; }
        public int _offset{ get; private set; }
        public int[] _shape { get; private set; } = new int[4];
        public int[] _stride { get; private set; } = new int[4];
        public TensorLayout()
        {
            _dtype = DType.Invalid;
            _ndim = 0;
            _offset = 0;
        }
        public TensorLayout(DType dtype, Span<int> shape)
        {
            _dtype = dtype;
            _offset = 0;
            InitContiguousLayout(shape);
        }
        internal void InitContiguousLayout()
        {
            int s = 1;
            for (int i = 0; i < _ndim; i++)
            {
                _stride[i] = s;
                s *= _shape[i];
            }
        }
        internal void InitContiguousLayout(Span<int> shape)
        {
            _ndim = shape.Length;
            if (_ndim > MAX_NDIM)
            {
                throw new InvalidLayoutException(shape);
            }
            for (int i = 0; i < _ndim; i++){
                _shape[_ndim - i - 1] = shape[i];
            }
            InitContiguousLayout();
        }

        public int total_elems(){
            if(_ndim == 0){
                return 0;
            }
            int res = 1;
            for (int i = 0; i < _ndim; i++){
                res *= _shape[i];
            }
            return res;
        }
    }
    

}