using NN.Native.Basic;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Extensions
{
    internal static class IndexOffsetConvertExtension
    {
        [method:MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe static int IndicesToOffset(this NativeLayout layout, int* indices)
        {
            int res = 0;
            for (int i = 0; i < layout._ndim; i++)
            {
                res += indices[i] * layout._stride[i];
            }
            return res;
        }
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe static void OffsetToIndices(this NativeLayout layout, int* indices, int offset)
        {
            for (int i = 0; i < layout._ndim; i++)
            {
                int idx = offset / layout._stride[i];
                indices[i] = idx;
                offset -= idx * layout._stride[i];
            }
        }
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe static int IndicesToOffset(this NativeLayout layout, int[] indices)
        {
            int res = 0;
            for (int i = 0; i < layout._ndim; i++)
            {
                res += indices[i] * layout._stride[i];
            }
            return res;
        }
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe static void OffsetToIndices(this NativeLayout layout, int[] indices, int offset)
        {
            for (int i = 0; i < layout._ndim; i++)
            {
                int idx = offset / layout._stride[i];
                indices[i] = idx;
                offset -= idx * layout._stride[i];
            }
        }
        public unsafe static string GetOffsetDescription(this NativeLayout layout, int offset)
        {
            var indices = new int[layout._ndim];
            layout.OffsetToIndices(indices, offset);
            return $"LayoutInfo: {layout}, CurrentIndex: ({string.Join(',', indices)})";
        }
    }
}
