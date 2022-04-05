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
    public class DimExceedException:Exception{
        protected string _message;
        public override string Message => _message;
        public DimExceedException(int dim){
            _message = $"Exceed max dim: the max is {Numnet.Tensor.Base.TensorShape.MAX_NDIM}, but {dim} is rerquired.";
        }
    }
}