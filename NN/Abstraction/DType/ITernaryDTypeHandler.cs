using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Abstraction.DType
{
    public interface ITernaryDTypeHandler<TA, TB, TC> where TA : unmanaged where TB : unmanaged where TC: unmanaged
    {
        /// <summary>
        /// dst = a + b
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="dst"></param>
        static abstract void Add(in TA a, in TB b, ref TC dst);
        /// <summary>
        /// dst = a - b
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="dst"></param>
        static abstract void Substract(in TA a, in TB b, ref TC dst);
        /// <summary>
        /// dst = a * b
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="dst"></param>
        static abstract void Multiply(in TA a, in TB b, ref TC dst);
        /// <summary>
        /// dst = a / b
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="dst"></param>
        static abstract void Divide(in TA a, in TB b, ref TC dst);
        /// <summary>
        /// dst = a % b
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="dst"></param>
        static abstract void Module(in TA a, in TB b, ref TC dst);
        /// <summary>
        /// dst += a * b
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="dst"></param>
        static abstract TC MultiplyAndAdd(TA a, TB b, TC dst);
    }
}
