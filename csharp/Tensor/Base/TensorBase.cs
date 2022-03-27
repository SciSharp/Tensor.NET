using System.Buffers;
using Numnet.Native;

namespace Numnet.Base{
    public abstract class TensorBase
    {
        internal unsafe delegate IntPtr DoubleInputOperation(NativeTensor* a, NativeTensor* b, NativeTensor* oup, IntPtr param, Provider provider);
        internal TensorMemory _dataHandle;
        internal TensorLayout _layout;
        protected void Pin(out MemoryHandle handle){
            _dataHandle.Pin(out handle);
        }
        unsafe internal static IntPtr Execute(TensorBase a, TensorBase b, TensorBase oup, DoubleInputOperation func, IntPtr param, Provider provider){
            MemoryHandle handleA, handleB, handleOup;
            a.Pin(out handleA);
            b.Pin(out handleB);
            oup.Pin(out handleOup);
            IntPtr status;
            Console.WriteLine("字符串地址= 0x{0:x}", (long)handleOup.Pointer);
            fixed(ulong* shapeAPtr = a._layout._shape, shapeBPtr = b._layout._shape, shapeOupPtr = oup._layout._shape, 
                        strideAPtr = a._layout._stride, strideBPtr = b._layout._stride, strideOupPtr = oup._layout._stride){
                NativeTensor nativeA = new NativeTensor()
                {
                    dtype = a._layout._dtype,
                    ndim = a._layout._ndim,
                    offset = a._layout._offset,
                    shape = new IntPtr(shapeAPtr),
                    stride = new IntPtr(strideAPtr),
                    data = new IntPtr(handleA.Pointer)
                };
                NativeTensor nativeB = new NativeTensor()
                {
                    dtype = b._layout._dtype,
                    ndim = b._layout._ndim,
                    offset = b._layout._offset,
                    shape = new IntPtr(shapeBPtr),
                    stride = new IntPtr(strideBPtr),
                    data = new IntPtr(handleB.Pointer)
                };
                NativeTensor nativeOup = new NativeTensor()
                {
                    dtype = oup._layout._dtype,
                    ndim = oup._layout._ndim,
                    offset = oup._layout._offset,
                    shape = new IntPtr(shapeOupPtr),
                    stride = new IntPtr(strideOupPtr),
                    data = new IntPtr(handleOup.Pointer)
                };
                Console.WriteLine(nativeA.ndim);
                Console.WriteLine(nativeB.ndim);
                Console.WriteLine(nativeOup.ndim);
                status = func(&nativeA, &nativeB, &nativeOup, param, provider);
            }
            handleA.Dispose();
            handleB.Dispose();
            handleOup.Dispose();
            return status;
        }
    }
}