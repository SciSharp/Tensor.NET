using Numnet.Native;

namespace Numnet.Base{
    internal sealed class TensorLayout
    {
        static readonly int MAX_NDIM = 4;
        public DType _dtype { get; private set; }
        public int _ndim{ get; private set; }
        public ulong _offset{ get; private set; }
        public ulong[] _shape { get; private set; } = new ulong[4];
        public ulong[] _stride { get; private set; } = new ulong[4];
        public TensorLayout()
        {
            _dtype = DType.Invalid;
            _ndim = 0;
            _offset = 0;
        }
        public TensorLayout(DType dtype, Span<ulong> shape)
        {
            _dtype = dtype;
            _offset = 0;
            InitContiguousLayout(shape);
        }
        public void InitContiguousLayout()
        {
            ulong s = 1;
            for (int i = 0; i < _ndim; i++)
            {
                _stride[i] = s;
                s *= _shape[i];
            }
        }
        public void InitContiguousLayout(Span<ulong> shape)
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

        public ulong total_elems(){
            if(_ndim == 0){
                return 0;
            }
            ulong res = 1;
            for (int i = 0; i < _ndim; i++){
                res *= _shape[i];
            }
            return res;
        }
    }
    

}