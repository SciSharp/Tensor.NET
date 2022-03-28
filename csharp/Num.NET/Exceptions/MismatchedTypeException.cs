namespace Numnet.Exceptions{
    public class MismatchedTypeException:NNRuntimeException{
        public MismatchedTypeException(IntPtr status):base(status){}
    }
}