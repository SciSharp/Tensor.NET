namespace Numnet.Exceptions{
    public class InvalidShapeException:Exception{
        protected string _message;
        public override string Message => _message;
        public InvalidShapeException(int[] shape, string method){
            _message = $"Shape [{string.Join(',', shape)}] is invalid for operation {method}.";
        }
        public InvalidShapeException(string info){
            _message = info;
        }
    }
}