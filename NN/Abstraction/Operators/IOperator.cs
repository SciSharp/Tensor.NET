using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Abstraction.Operators
{
    public interface IOperator
    {
#if NET7_0_OR_GREATER
        static abstract bool IsThreadSafe { get; }
        static abstract OperatorHandlerType HandlerType { get; }
#else
        bool IsThreadSafe { get; }
        OperatorHandlerType HandlerType { get; }
#endif
    }
}
