using Tensornet.Common;

namespace Tensornet.Exceptions{
    public class MismatchedShapeException:Exception{
        protected string _message;
        public override string Message => _message;
        public MismatchedShapeException(string message){
            _message = message;
        }
        public MismatchedShapeException(TensorShape lhs, TensorShape rhs, string methodName){
            _message = $"Tensor shape mismatched in [{methodName}], one is {lhs.ToString()}, the other is {rhs.ToString()}";
        }
    }
}