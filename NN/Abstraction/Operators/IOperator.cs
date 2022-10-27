using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Abstraction.Operators
{
    public interface IOperator
    {
        static abstract bool IsThreadSafe { get; }
        static abstract OperatorHandlerType HandlerType { get; }
    }
}
