namespace Numnet.Exceptions{
    public class MismatchedShapeException:NNRuntimeException{
        public MismatchedShapeException(IntPtr status):base(status){}
    }
}