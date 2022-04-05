using System.Buffers;
using Numnet.Native;

namespace Numnet.Tensor.Base{
    public abstract class TensorBase:ITensor
    {
        internal unsafe delegate IntPtr DoubleInputOperation(NativeTensor* a, NativeTensor* b, NativeTensor* oup, IntPtr param, Provider provider);
        internal unsafe delegate IntPtr SingleInputOperation(NativeTensor* inp, NativeTensor* oup, IntPtr param, Provider provider);
        public TensorLayout TLayout{get; protected set; }
        protected abstract void Pin(out MemoryHandle handle);
        public TensorBase(TensorLayout layout){
            TLayout = layout;
        }
        unsafe internal static IntPtr Execute(TensorBase a, TensorBase b, TensorBase oup, DoubleInputOperation func, IntPtr param, Provider provider){
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

        unsafe internal static IntPtr Execute(TensorBase inp, TensorBase oup, SingleInputOperation func, IntPtr param, Provider provider){
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
    }
}