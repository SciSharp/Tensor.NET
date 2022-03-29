using Numnet.Native;
using System.Text;

namespace Numnet.Tensor.Base{
    public sealed class TensorLayout
    {
        static readonly int MAX_NDIM = 4;
        public DType DType { get; internal set; }
        public int NDim{ get; internal set; }
        public int Offset{ get; internal set; }
        public int[] Shape { get; internal set; } = new int[4];
        public int[] Stride { get; internal set; } = new int[4];
        public TensorLayout()
        {
            DType = DType.Invalid;
            NDim = 0;
            Offset = 0;
        }
        public TensorLayout(DType dtype, Span<int> shape)
        {
            DType = dtype;
            Offset = 0;
            InitContiguousLayout(shape);
        }
        internal void InitContiguousLayout()
        {
            int s = 1;
            for (int i = 0; i < NDim; i++)
            {
                Stride[i] = s;
                s *= Shape[i];
            }
        }
        internal void InitContiguousLayout(Span<int> shape)
        {
            NDim = shape.Length;
            if (NDim > MAX_NDIM)
            {
                throw new InvalidLayoutException(shape);
            }
            for (int i = 0; i < NDim; i++){
                Shape[NDim - i - 1] = shape[i];
            }
            InitContiguousLayout();
        }

        public int TotalElemCount(){
            if(NDim == 0){
                return 0;
            }
            int res = 1;
            for (int i = 0; i < NDim; i++){
                res *= Shape[i];
            }
            return res;
        }

        public override string ToString()
        {
            return $"TensorLayout({GetInfoString()})";
        }

        internal string GetInfoString(){
            StringBuilder r = new StringBuilder();
            if (NDim == 0) {
                r.Append(" Scalar");
            } else {
                r.Append("shape = {");
                for (int i = 0; i < NDim; i++) {
                    r.Append(Shape[NDim - 1 - i]);
                    if (i != NDim - 1) r.Append(", ");
                }
                r.Append("}");
            }
            r.Append(", dtype = ");
            r.Append(DType);
            return r.ToString();
        }
    }
    

}