using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Extensions
{
    internal static class SpanExtension
    {
        public static void Print2D<T>(this ReadOnlySpan<T> span, int cols)
        {
            for(int i = 0; i < span.Length / cols; i++)
            {
                for(int j = 0; j < cols; j++)
                {
                    Console.Write(span[i * cols + j] + ", ");
                }
                Console.WriteLine();
            }
        }
        public static unsafe void Print2D<T>(this Span<T> span, int cols)
        {
            for (int i = 0; i < span.Length / cols; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    Console.Write(span[i * cols + j] + ", ");
                }
                Console.WriteLine();
            }
        }
    }
}
