using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Abstraction.DType
{
    public interface ITernaryDTypeHandler<T> where T : unmanaged
    {
        /// <summary>
        /// a + b
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        T Add(T a, T b);
        /// <summary>
        /// a - b
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        T Subtract(T a, T b);
        /// <summary>
        /// a * b
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        T Multiply(T a, T b);
        /// <summary>
        /// a / b
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        T Divide(T a, T b);
        /// <summary>
        /// a % b
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        T Module(T a, T b);
        /// <summary>
        /// a * b + c
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="dst"></param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        T MultiplyAndAdd(T a, T b, T c);
    }
}
