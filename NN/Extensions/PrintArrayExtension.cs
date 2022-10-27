using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NN.Native.Basic;
using NN.Native.Data;

namespace NN.Native.Extensions
{
    public static class PrintArrayExtension
    {
        public static string Print<T>(this NativeArray<T> array) where T: unmanaged
        {
            var layout = array._layout;
            var indexEnumerator = NativeLayout.GetIndexEnumerator(layout);

            StringBuilder r = new StringBuilder($"NativeArray {layout}:\n");
            var span = array.Span;
            for (int i = 0; i < layout.TotalElemCount(); i++)
            {
                int mod = 1;
                for (int j = layout._ndim - 1; j >= 0; j--)
                {
                    mod *= layout._shape[j];
                    if (i % mod == 0)
                    {
                        r.Append("[");
                    }
                    else
                    {
                        break;
                    }
                }
                r.Append(" ").Append(span[indexEnumerator.MoveNext()]);

                if ((i + 1) % layout._shape[layout._ndim - 1] != 0) r.Append(",");

                r.Append(" ");
                mod = 1;
                int hit_times = 0;
                for (int j = layout._ndim - 1; j >= 0; j--)
                {
                    mod *= layout._shape[j];
                    if ((i + 1) % mod == 0)
                    {
                        r.Append("]");
                        hit_times++;
                    }
                    else
                    {
                        break;
                    }
                }
                if (hit_times > 0 && hit_times < layout._ndim)
                {
                    r.Append(",\n");
                    for (int j = 0; j < layout._ndim - hit_times; j++)
                    {
                        r.Append(" ");
                    }
                }
            }
            // r.Append("\n");
            return r.ToString();
        }
    }
}
