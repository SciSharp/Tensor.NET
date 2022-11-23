using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Abstraction.DType
{
    public unsafe interface IBinaryDTypeHandler<T> where T : unmanaged
    {
        bool AreEqual (T a, T b);
        /// <summary>
        /// Return if a is greater than b.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        bool IsGraeter(T a, T b);
        /// <summary>
        /// Return if a is less than b.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        bool IsLess(T a, T b);

        T GetNegative(T x);
    }
}
