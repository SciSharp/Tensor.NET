using System.Buffers;
using Numnet.Native;
using Numnet.Common;

namespace Numnet.Base{
    public abstract class TensorBase:ITensor
    {
        internal unsafe delegate IntPtr DoubleInputOperation(NativeTensor* a, NativeTensor* b, NativeTensor* oup, IntPtr param, Provider provider);
        public TensorLayout TLayout{get; protected set; }
        protected abstract void Pin(out MemoryHandle handle);
        unsafe internal static IntPtr Execute(TensorBase a, TensorBase b, TensorBase oup, DoubleInputOperation func, IntPtr param, Provider provider){
            MemoryHandle handleA, handleB, handleOup;
            a.Pin(out handleA);
            b.Pin(out handleB);
            oup.Pin(out handleOup);
            IntPtr status;
            Console.WriteLine("字符串地址= 0x{0:x}", (long)handleOup.Pointer);
            fixed(ulong* shapeAPtr = a.TLayout._shape, shapeBPtr = b.TLayout._shape, shapeOupPtr = oup.TLayout._shape, 
                        strideAPtr = a.TLayout._stride, strideBPtr = b.TLayout._stride, strideOupPtr = oup.TLayout._stride){
                NativeTensor nativeA = new NativeTensor()
                {
                    dtype = a.TLayout._dtype,
                    ndim = a.TLayout._ndim,
                    offset = a.TLayout._offset,
                    shape = new IntPtr(shapeAPtr),
                    stride = new IntPtr(strideAPtr),
                    data = new IntPtr(handleA.Pointer)
                };
                NativeTensor nativeB = new NativeTensor()
                {
                    dtype = b.TLayout._dtype,
                    ndim = b.TLayout._ndim,
                    offset = b.TLayout._offset,
                    shape = new IntPtr(shapeBPtr),
                    stride = new IntPtr(strideBPtr),
                    data = new IntPtr(handleB.Pointer)
                };
                NativeTensor nativeOup = new NativeTensor()
                {
                    dtype = oup.TLayout._dtype,
                    ndim = oup.TLayout._ndim,
                    offset = oup.TLayout._offset,
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