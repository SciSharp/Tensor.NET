using System.Buffers;
using Numnet.Native;
using System.Text;
using Numnet.Common;
using Numnet.Exceptions;
using System.Runtime.InteropServices;

namespace Numnet{
    public partial class Tensor
    {
        internal delegate IntPtr DoubleInputOperation(IntPtr a, IntPtr b, IntPtr oup, IntPtr param, NativeProvider provider);
        internal delegate IntPtr SingleInputOperation(IntPtr inp, IntPtr oup, IntPtr param, NativeProvider provider);
        internal TensorLayout TLayout{get; set; }
        internal TensorMemory TMemory{ get; set; }
        public Span<int> Shape{
            get{
                return TLayout.Shape.AsSpan(0, TLayout.NDim);
            }
        }
        public int Dim{get { return TLayout.NDim; } }
        public DType DataType{get { return TLayout.DType; } }
        public static NativeProvider Provider { get; set; } = NativeProvider.Naive;
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
        internal Tensor(TensorShape shape, DType dtype){
            TLayout = new TensorLayout(shape, dtype);
            TMemory = new TensorMemory(shape.TotalElemCount(), dtype);
        }
        internal int IndicesToPosition(params int[] indices){
            if(indices.Length != TLayout.NDim){
                throw new InvalidArgumentException($"Index does not have same dims with tensor, " + 
                    "the index is {indices.Length} dims but the tensor is {TLayout.NDim}.");
            }
            int res = 0;
            for (int i = 0; i < TLayout.NDim; i++) {
                if(indices[i] < -1 || indices[i] >= TLayout.Shape[i]){
                    throw new InvalidArgumentException($"{i}th index exceeds the bound of the shape. Index is {indices[i]}, limit is {TLayout.Shape[i]}.");
                }
                res += indices[i] * TLayout.Stride[i];
            }
            return res + TLayout.Offset;
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
        internal void SetValue(int idx, ValueType value){
            if(TLayout.DType == DType.Int32){
                AsSpan<int>()[idx] = (int)value;
            }
            else if(TLayout.DType == DType.Int64){
                AsSpan<long>()[idx] = (long)value;
            }
            else if(TLayout.DType == DType.Float32){
                AsSpan<float>()[idx] = (float)value;
            }
            else if(TLayout.DType == DType.Float64){
                AsSpan<double>()[idx] = (double)value;
            }
            else if(TLayout.DType == DType.Bool){
                AsSpan<bool>()[idx] = (bool)value;
            }
            else{
                throw new NotImplementedException();
            }
        }
        public Span<T> AsSpan<T>() where T:struct{
            return TMemory.AsSpan<T>();
        }
        public T At<T>(params int[] indices) where T:struct{
            var targetTypeInfo = TensorTypeInfo.GetTypeInfo(typeof(T));
            var currentTypeInfo = TensorTypeInfo.GetTypeInfo(TensorTypeInfo.GetTypeInfo(TLayout.DType));
            if(targetTypeInfo._dtype == TLayout.DType) 
                return AsSpan<T>()[IndicesToPosition(indices)];
            if(targetTypeInfo._dtype == DType.Bool)
                throw new NotImplementedException("The conversion to bool is not suppoted so far. You can get the value and convert it yourself as an alternative.");
            if(targetTypeInfo._priority >= currentTypeInfo._priority)
                return (T)this[indices];
            throw new MismatchedTypeException($"The conversion from {Enum.GetName(typeof(DType), TLayout.DType)} " +
                        $"to {Enum.GetName(typeof(DType), targetTypeInfo._dtype)} has risk of loss. " +
                        $"If you really need this operation, please use indexer to get the value and then cast it yourself.");
        }
        /// <summary>
        /// Treat the current Tensor as Tensor<T>. The returned Tensor will share the memory with the current Tensor.
        /// Note that this operation is not a type conversion. If the generic type T does not match the DataType of the Tensor, 
        /// an exception will be thrown. If you need a type conversion , please use ConvertTo or ConvertTo<T>.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public Tensor<T> AsTensor<T>() where T : struct{
            return new Tensor<T>(TMemory, TLayout);
        }

        public Tensor<T> ConvertTo<T>() where T : struct{
            throw new NotImplementedException();
        }

        public Tensor ConvertTo(DType targetType){
            throw new NotImplementedException();
        }
        public bool IsType<T>(){
            return TLayout.DType == TensorTypeInfo.GetTypeInfo(typeof(T))._dtype;
        }
        public bool IsType(DType dtype){
            return TLayout.DType == dtype;
        }
        public ValueType this[params int[] index]{
            get{
                return GetValue(IndicesToPosition(index));
            }
            set{
                SetValue(IndicesToPosition(index), value);
            }
        }
        unsafe internal static IntPtr Execute(DoubleInputOperation func, TensorMemory a, TensorMemory b, TensorMemory oup, 
                            TensorLayout layoutA, TensorLayout layoutB, TensorLayout layoutOup, IntPtr param, NativeProvider provider){
            MemoryHandle handleA, handleB, handleOup;
            a.Pin(out handleA);
            b.Pin(out handleB);
            oup.Pin(out handleOup);
            IntPtr status;
            fixed(int* shapeAPtr = layoutA.Shape, shapeBPtr = layoutB.Shape, shapeOupPtr = layoutOup.Shape, 
                        strideAPtr = layoutA.Stride, strideBPtr = layoutB.Stride, strideOupPtr = layoutOup.Stride){
                NativeTensor nativeA = new NativeTensor()
                {
                    dtype = layoutA.DType,
                    ndim = layoutA.NDim,
                    offset = layoutA.Offset,
                    shape = new IntPtr(shapeAPtr),
                    stride = new IntPtr(strideAPtr),
                    data = new IntPtr(handleA.Pointer)
                };
                NativeTensor nativeB = new NativeTensor()
                {
                    dtype = layoutB.DType,
                    ndim = layoutB.NDim,
                    offset = layoutB.Offset,
                    shape = new IntPtr(shapeBPtr),
                    stride = new IntPtr(strideBPtr),
                    data = new IntPtr(handleB.Pointer)
                };
                NativeTensor nativeOup = new NativeTensor()
                {
                    dtype = layoutOup.DType,
                    ndim = layoutOup.NDim,
                    offset = layoutOup.Offset,
                    shape = new IntPtr(shapeOupPtr),
                    stride = new IntPtr(strideOupPtr),
                    data = new IntPtr(handleOup.Pointer)
                };
                status = func(new IntPtr(&nativeA), new IntPtr(&nativeB), new IntPtr(&nativeOup), param, provider);
            }
            handleA.Dispose();
            handleB.Dispose();
            handleOup.Dispose();
            return status;
        }

        unsafe internal static IntPtr Execute(SingleInputOperation func, TensorMemory inp, TensorMemory oup, TensorLayout layoutInp, 
                                                    TensorLayout layoutOup, IntPtr param, NativeProvider provider){
            MemoryHandle handleInp, handleOup;
            inp.Pin(out handleInp);
            oup.Pin(out handleOup);
            IntPtr status;
            fixed(int* shapeInpPtr = layoutInp.Shape, shapeOupPtr = layoutOup.Shape, 
                        strideInpPtr = layoutInp.Stride, strideOupPtr = layoutOup.Stride){
                NativeTensor nativeInp = new NativeTensor()
                {
                    dtype = layoutInp.DType,
                    ndim = layoutInp.NDim,
                    offset = layoutInp.Offset,
                    shape = new IntPtr(shapeInpPtr),
                    stride = new IntPtr(strideInpPtr),
                    data = new IntPtr(handleInp.Pointer)
                };
                NativeTensor nativeOup = new NativeTensor()
                {
                    dtype = layoutOup.DType,
                    ndim = layoutOup.NDim,
                    offset = layoutOup.Offset,
                    shape = new IntPtr(shapeOupPtr),
                    stride = new IntPtr(strideOupPtr),
                    data = new IntPtr(handleOup.Pointer)
                };
                status = func(new IntPtr(&nativeInp), new IntPtr(&nativeOup), param, provider);
            }
            handleInp.Dispose();
            handleOup.Dispose();
            return status;
        }

        public override string ToString()
        {
            // Somtimes the tensor is not contiguous, so we need to convert the index calculated 
            // by shape to the real index calculated by sride.
            Func<int, int> getRealPos = idx => {
                int res = 0;
                int mod = 1;
                for (int i = TLayout.NDim - 1; i >= 1; i--) mod *= TLayout.Shape[i];
                for (int i = 0; i < TLayout.NDim; i++) {
                    int shape = idx / mod;
                    idx -= shape * mod;
                    res += shape * TLayout.Stride[i];
                    if(i < TLayout.NDim - 1 ) mod /= TLayout.Shape[i + 1];
                }
                return res + TLayout.Offset;
            };

            StringBuilder r = new StringBuilder($"Tensor({TLayout.GetInfoString()}):\n");
            for (int i = 0; i < TLayout.TotalElemCount(); i++) {
                int mod = 1;
                for (int j = TLayout.NDim - 1; j >= 0; j--) {
                    mod *= TLayout.Shape[j];
                    if (i % mod == 0) {
                        r.Append("[");
                    } else {
                        break;
                    }
                }
                r.Append(" ").Append(GetValue(getRealPos(i)));

                if ((i + 1) % TLayout.Shape[TLayout.NDim - 1] != 0) r.Append(",");

                r.Append(" ");
                mod = 1;
                int hit_times = 0;
                for (int j = TLayout.NDim - 1; j >= 0; j--) {
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

        public new T this[params int[] index]{
            get{
                return AsSpan<T>()[IndicesToPosition(index)];
            }
            set{
                AsSpan<T>()[IndicesToPosition(index)] = value;
            }
        }
    }
}