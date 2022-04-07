using System.Buffers;
using Numnet.Native;
using System.Text;
using Numnet.Common;

namespace Numnet{
    public partial class Tensor
    {
        internal unsafe delegate IntPtr DoubleInputOperation(NativeTensor* a, NativeTensor* b, NativeTensor* oup, IntPtr param, Provider provider);
        internal unsafe delegate IntPtr SingleInputOperation(NativeTensor* inp, NativeTensor* oup, IntPtr param, Provider provider);
        public TensorLayout TLayout{get; protected set; }
        internal TensorMemory TMemory{ get; set; }
        protected void Pin(out MemoryHandle handle){
            TMemory.Pin(out handle);
        }
        internal Tensor(TensorMemory memory, TensorLayout layout){
            TMemory = memory;
            TLayout = layout;
        }
        internal Tensor(TensorLayout layout){
            TLayout = layout;
            TMemory = new TensorMemory(layout.TotalElemCount(), layout.DType);
        }
        internal ValueType GetValue(int idx){
            if(TLayout.DType == DType.Int32){
                return AsSpan<int>()[idx];
            }
            else if(TLayout.DType == DType.Int64){
                return AsSpan<long>()[idx];
            }
            else if(TLayout.DType == DType.Float32){
                return AsSpan<float>()[idx];
            }
            else if(TLayout.DType == DType.Float64){
                return AsSpan<double>()[idx];
            }
            else if(TLayout.DType == DType.Bool){
                return AsSpan<bool>()[idx];
            }
            else{
                throw new NotImplementedException();
            }
        }
        public Span<T> AsSpan<T>() where T:struct{
            return TMemory.AsSpan<T>();
        }
        public Tensor<T> To<T>() where T:struct{
            if(TensorTypeInfo.GetTypeInfo(typeof(T))._dtype != TLayout.DType){
                throw new NotImplementedException();
            }
            return new Tensor<T>(this);
        }
        public bool IsType<T>(){
            return TLayout.DType == TensorTypeInfo.GetTypeInfo(typeof(T))._dtype;
        }
        public bool IsType(DType dtype){
            return TLayout.DType == dtype;
        }
        unsafe internal static IntPtr Execute(Tensor a, Tensor b, Tensor oup, DoubleInputOperation func, IntPtr param, Provider provider){
            MemoryHandle handleA, handleB, handleOup;
            a.Pin(out handleA);
            b.Pin(out handleB);
            oup.Pin(out handleOup);
            IntPtr status;
            fixed(int* shapeAPtr = a.TLayout.Shape, shapeBPtr = b.TLayout.Shape, shapeOupPtr = oup.TLayout.Shape, 
                        strideAPtr = a.TLayout.Stride, strideBPtr = b.TLayout.Stride, strideOupPtr = oup.TLayout.Stride){
                NativeTensor nativeA = new NativeTensor()
                {
                    dtype = a.TLayout.DType,
                    ndim = a.TLayout.NDim,
                    offset = a.TLayout.Offset,
                    shape = new IntPtr(shapeAPtr),
                    stride = new IntPtr(strideAPtr),
                    data = new IntPtr(handleA.Pointer)
                };
                NativeTensor nativeB = new NativeTensor()
                {
                    dtype = b.TLayout.DType,
                    ndim = b.TLayout.NDim,
                    offset = b.TLayout.Offset,
                    shape = new IntPtr(shapeBPtr),
                    stride = new IntPtr(strideBPtr),
                    data = new IntPtr(handleB.Pointer)
                };
                NativeTensor nativeOup = new NativeTensor()
                {
                    dtype = oup.TLayout.DType,
                    ndim = oup.TLayout.NDim,
                    offset = oup.TLayout.Offset,
                    shape = new IntPtr(shapeOupPtr),
                    stride = new IntPtr(strideOupPtr),
                    data = new IntPtr(handleOup.Pointer)
                };
                status = func(&nativeA, &nativeB, &nativeOup, param, provider);
            }
            handleA.Dispose();
            handleB.Dispose();
            handleOup.Dispose();
            return status;
        }

        unsafe internal static IntPtr Execute(Tensor inp, Tensor oup, SingleInputOperation func, IntPtr param, Provider provider){
            MemoryHandle handleInp, handleOup;
            inp.Pin(out handleInp);
            oup.Pin(out handleOup);
            IntPtr status;
            fixed(int* shapeInpPtr = inp.TLayout.Shape, shapeOupPtr = oup.TLayout.Shape, 
                        strideInpPtr = inp.TLayout.Stride, strideOupPtr = oup.TLayout.Stride){
                NativeTensor nativeInp = new NativeTensor()
                {
                    dtype = inp.TLayout.DType,
                    ndim = inp.TLayout.NDim,
                    offset = inp.TLayout.Offset,
                    shape = new IntPtr(shapeInpPtr),
                    stride = new IntPtr(strideInpPtr),
                    data = new IntPtr(handleInp.Pointer)
                };
                NativeTensor nativeOup = new NativeTensor()
                {
                    dtype = oup.TLayout.DType,
                    ndim = oup.TLayout.NDim,
                    offset = oup.TLayout.Offset,
                    shape = new IntPtr(shapeOupPtr),
                    stride = new IntPtr(strideOupPtr),
                    data = new IntPtr(handleOup.Pointer)
                };
                status = func(&nativeInp, &nativeOup, param, provider);
            }
            handleInp.Dispose();
            handleOup.Dispose();
            return status;
        }

        public override string ToString()
        {
            Func<int, int> getRealPos = idx => {
                int res = 0;
                for (int i =  TLayout.NDim - 1; i >= 0; i--) {
                    int mod = TLayout.Stride[i];
                    if (mod <= 0){
                        int j = i - 1;
                        while(j >= 0 && TLayout.Stride[j] <= 0) j--;
                        if(j < 0){
                            mod = 1;
                        }
                        else{
                            mod = TLayout.Stride[j] * TLayout.Shape[j];
                        }
                    }
                    else
                        res += idx / mod * mod;
                    idx %= mod;
                    if (idx <= 0) break;
                }
                return res;
            };

            StringBuilder r = new StringBuilder($"Tensor({TLayout.GetInfoString()}):\n");
            for (int i = 0; i < TLayout.TotalElemCount(); i++) {
                int mod = 1;
                for (int j = 0; j < TLayout.NDim; j++) {
                    mod *= TLayout.Shape[j];
                    if (i % mod == 0) {
                        r.Append("[");
                    } else {
                        break;
                    }
                }
                r.Append(" ").Append(GetValue(getRealPos(i)));

                if ((i + 1) % TLayout.Shape[0] != 0) r.Append(",");

                r.Append(" ");
                mod = 1;
                int hit_times = 0;
                for (int j = 0; j < TLayout.NDim; j++) {
                    mod *= TLayout.Shape[j];
                    if ((i + 1) % mod == 0) {
                        r.Append("]");
                        hit_times++;
                    } else {
                        break;
                    }
                }
                if (hit_times > 0 && hit_times < TLayout.NDim) {
                    r.Append(",\n");
                    for (int j = 0; j < TLayout.NDim - hit_times; j++) {
                        r.Append(" ");
                    }
                }
            }
            // r.Append("\n");
            return r.ToString();
        }
    }

    public sealed partial class Tensor<T>: Tensor where T:struct{
        internal Tensor(TensorMemory memory, TensorLayout layout):base(memory, layout){ }
        internal Tensor(TensorLayout layout) : base(layout) { }

        internal Tensor(Tensor source):base(source.TMemory, source.TLayout){

        }
        // public Tensor(IEnumerable<T> data, Span<int> shape):base(new TensorLayout(TensorTypeInfo.GetTypeInfo(typeof(T))._dtype, shape)){
        //     TMemory = new TensorMemory<T>(data.ToArray());
        // }

        // public Tensor(T[] data, Span<int> shape):base(new TensorLayout(TensorTypeInfo.GetTypeInfo(typeof(T))._dtype, shape)){
        //     if(data.Length < TLayout.TotalElemCount()){
        //         // TODO
        //         throw new Exception();
        //     }
        //     TMemory = new TensorMemory<T>(data);
        // }

        // public Tensor(Array data):base(new TensorLayout()){
        //     var ndim = data.Rank;
        //     int[] shape = new int[ndim];
        //     for (int i = 0; i < ndim; i++){
        //         shape[i] = data.GetLength(i);
        //     }
        //     var dtypeInfo = TensorTypeInfo.GetTypeInfo(data.GetType().GetElementType()!);
        //     TLayout.NDim = ndim;
        //     TLayout.DType = dtypeInfo._dtype;
        //     TLayout.Shape = shape;
        //     TLayout.InitContiguousLayout();
        //     TMemory = new TensorMemory<T>(data);
        // }

        // internal Tensor(int[] shape):base(new TensorLayout(TensorTypeInfo.GetTypeInfo(typeof(T))._dtype, shape)){
        //     int length = 1;
        //     foreach(var s in shape){
        //         length *= s;
        //     }
        //     TMemory = new TensorMemory<T>(length);
        // }

        // internal Tensor(TensorMemory<T> memory, TensorLayout layout):base(layout){
        //     if(TensorTypeInfo.GetTypeInfo(typeof(T))._dtype != layout.DType){
        //         throw new NotImplementedException();
        //     }
        //     TMemory = memory;
        // }
    }
}