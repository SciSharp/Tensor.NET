using Tensornet.Common;
using Tensornet.Native;
using Tensornet.Exceptions;
using Tensornet.Native.Param;
using System.Buffers;
using System.Runtime.InteropServices;

namespace Tensornet{
    public static partial class Tensor{

        public static Tensor<T> Concat<T>(IEnumerable<Tensor<T>> src, int axis) where T : struct, IEquatable<T>, IConvertible
        {
            Tensor<T> res = new Tensor<T>(ConcatDeduceLayout(src, axis));
            res.TLayout.InitContiguousLayout();
            ConcatInternal(src.Select(x => x.TMemory).ToArray(), src.Select(x => x.TLayout).ToArray(), res, axis);
            return res;
        }
        private unsafe static void ConcatInternal<T>(TensorMemory<T>[] srcMemory, TensorLayout[] srcLayout, Tensor<T> dst, int axis) where T : struct, IEquatable<T>, IConvertible{
            MemoryHandle[] srcHandles = new MemoryHandle[srcMemory.Length];
            MemoryHandle dstHandle;
            dst.TMemory.Pin(out dstHandle);
            IntPtr[] srcPtrs = srcHandles.Select(x => new IntPtr(x.Pointer)).ToArray();
            GCHandle[] shapeHandles = new GCHandle[srcMemory.Length];
            GCHandle[] strideHandles = new GCHandle[srcMemory.Length];
            NativeTensor[] nativeTensors = new NativeTensor[srcMemory.Length];
            for (int i = 0; i < srcMemory.Length; i++){
                shapeHandles[i] = GCHandle.Alloc(srcLayout[i].Shape, GCHandleType.Pinned);
                strideHandles[i] = GCHandle.Alloc(srcLayout[i].Stride, GCHandleType.Pinned);
                srcMemory[i].Pin(out srcHandles[i]);
                nativeTensors[i] = new NativeTensor()
                {
                    dtype = srcLayout[i].DType,
                    ndim = srcLayout[i].NDim,
                    offset = srcLayout[i].Offset * TensorTypeInfo.GetTypeSize(srcLayout[i].DType),
                    shape = shapeHandles[i].AddrOfPinnedObject(),
                    stride = strideHandles[i].AddrOfPinnedObject(),
                    data = new IntPtr(srcHandles[i].Pointer)
                };
            }
            IntPtr status;
            ConcatParam param = new ConcatParam() { axis = axis };
            fixed (int* shapeOupPtr = dst.TLayout.Shape, strideOupPtr = dst.TLayout.Stride){
                NativeTensor nativeOup = new NativeTensor(){
                    dtype = dst.TLayout.DType,
                    ndim = dst.TLayout.NDim,
                    offset = dst.TLayout.Offset * TensorTypeInfo.GetTypeSize(dst.TLayout.DType),
                    shape = new IntPtr(shapeOupPtr),
                    stride = new IntPtr(strideOupPtr),
                    data = new IntPtr(dstHandle.Pointer)
                };
                fixed(NativeTensor* nativeTensorsPtr = nativeTensors){
                    status = NativeApi.Concat(new IntPtr(nativeTensorsPtr), nativeTensors.Length, new IntPtr(&nativeOup), new IntPtr(&param), Tensor<T>.Provider);
                }
            }
            NativeStatus.AssertOK(status);

            dstHandle.Dispose();
            for (int i = 0; i < shapeHandles.Length; i++){
                shapeHandles[i].Free();
                strideHandles[i].Free();
            }
        }
        private static TensorLayout ConcatDeduceLayout<T>(IEnumerable<Tensor<T>> src, int axis) where T : struct, IEquatable<T>, IConvertible{
            TensorLayout res = new TensorLayout();
            int n = src.Count();
            if (n < 2) {
                throw new InvalidArgumentException("At least two tensor should be srcutted to execute the concat op.");
            }
            var enumerator = src.GetEnumerator();
            enumerator.MoveNext();
            int ndim = enumerator.Current.TLayout.NDim;
            int sizeOnAxis = enumerator.Current.TLayout.Shape[axis];
            res.DType = enumerator.Current.TLayout.DType;
            for (int i = 0; i < ndim; i++) {
                res.Shape[i] = enumerator.Current.TLayout.Shape[i];
            }
            while(enumerator.MoveNext()) {
                if (enumerator.Current.TLayout.NDim != ndim) {
                    throw new InvalidArgumentException("The tensors to concat must have the same ndim.");
                }
                for (int j = 0; j < ndim; j++) {
                    if (j != axis &&
                        enumerator.Current.TLayout.Shape[j] != enumerator.Current.TLayout.Shape[j]) {
                        throw new InvalidArgumentException("Tensors to concat should have the same shape except the specified axis.");
                    }
                }
                sizeOnAxis += enumerator.Current.TLayout.Shape[axis];
            }
            if (axis < 0 || axis >= ndim) {
                throw new InvalidParamException("The specified axis to concat exceeds the max ndim of the tensors.");
            }
            res.NDim = ndim;
            res.Shape[axis] = sizeOnAxis;
            return res;
        }
    }
}