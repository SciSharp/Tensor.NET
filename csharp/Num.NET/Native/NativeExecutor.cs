using System.Buffers;
using Numnet.Common;

namespace Numnet.Native{
    internal class NativeExecutor{
        internal delegate IntPtr DoubleInputOperation(IntPtr a, IntPtr b, IntPtr oup, IntPtr param, NativeProvider provider);
        internal delegate IntPtr SingleInputOperation(IntPtr inp, IntPtr oup, IntPtr param, NativeProvider provider);
        internal delegate IntPtr SelfModifyOperation(IntPtr t, IntPtr param, NativeProvider provider);
        private NativeExecutor() { }
        unsafe internal static IntPtr Execute(DoubleInputOperation func, ITensorMemory a, ITensorMemory b, ITensorMemory oup, 
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
                    offset = layoutA.Offset * TensorTypeInfo.GetTypeSize(layoutA.DType),
                    shape = new IntPtr(shapeAPtr),
                    stride = new IntPtr(strideAPtr),
                    data = new IntPtr(handleA.Pointer)
                };
                NativeTensor nativeB = new NativeTensor()
                {
                    dtype = layoutB.DType,
                    ndim = layoutB.NDim,
                    offset = layoutB.Offset * TensorTypeInfo.GetTypeSize(layoutB.DType),
                    shape = new IntPtr(shapeBPtr),
                    stride = new IntPtr(strideBPtr),
                    data = new IntPtr(handleB.Pointer)
                };
                NativeTensor nativeOup = new NativeTensor()
                {
                    dtype = layoutOup.DType,
                    ndim = layoutOup.NDim,
                    offset = layoutOup.Offset * TensorTypeInfo.GetTypeSize(layoutOup.DType),
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

        unsafe internal static IntPtr Execute(SingleInputOperation func, ITensorMemory inp, ITensorMemory oup, TensorLayout layoutInp, 
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
                    offset = layoutInp.Offset * TensorTypeInfo.GetTypeSize(layoutInp.DType),
                    shape = new IntPtr(shapeInpPtr),
                    stride = new IntPtr(strideInpPtr),
                    data = new IntPtr(handleInp.Pointer)
                };
                NativeTensor nativeOup = new NativeTensor()
                {
                    dtype = layoutOup.DType,
                    ndim = layoutOup.NDim,
                    offset = layoutOup.Offset * TensorTypeInfo.GetTypeSize(layoutOup.DType),
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

        unsafe internal static IntPtr Execute(SelfModifyOperation func, ITensorMemory tm, TensorLayout layout, IntPtr param, NativeProvider provider){
            MemoryHandle mhandle;
            tm.Pin(out mhandle);
            IntPtr status;
            fixed(int* shapeInpPtr = layout.Shape, shapeOupPtr = layout.Shape, 
                        strideInpPtr = layout.Stride, strideOupPtr = layout.Stride){
                NativeTensor nativeTensor = new NativeTensor()
                {
                    dtype = layout.DType,
                    ndim = layout.NDim,
                    offset = layout.Offset * TensorTypeInfo.GetTypeSize(layout.DType),
                    shape = new IntPtr(shapeInpPtr),
                    stride = new IntPtr(strideInpPtr),
                    data = new IntPtr(mhandle.Pointer)
                };
                status = func(new IntPtr(&nativeTensor), param, provider);
            }
            mhandle.Dispose();
            return status;
        }
    }
}