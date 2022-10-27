using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Abstraction.DType
{
    public unsafe interface IBinaryDTypeHandler<TA, TB> where TA : unmanaged where TB : unmanaged
    {
        static abstract void AddInplace(ref TA src, in TB addend);
        static abstract void SubstractInplace(ref TA src, in TB subtrahend);
        static abstract void MultiplyInplace(ref TA src, in TB multiplicator);
        static abstract void DivideInplace(ref TA src, in TB divisor);
        static abstract void ModuleInplace(ref TA src, in TB modulus);
        //static abstract void LeftShift(ref TA src, ref TB dst, int offset);
        //static abstract void RightShift(ref TA src, ref TB dst, int offset);
        static abstract bool AreEqual (in TA a, in TB b);
        /// <summary>
        /// Return if a is greater than b.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        static abstract bool IsGraeter(in TA a, in TB b);
        /// <summary>
        /// Return if a is less than b.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        static abstract bool IsLess(in TA a, in TB b);
    }
}
