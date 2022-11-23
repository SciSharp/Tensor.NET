using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Core.Exceptions
{
    public class MismatchedTypeException: Exception
    {
        public MismatchedTypeException():base(){}
        public MismatchedTypeException(string message):base(message){}
    }
}
