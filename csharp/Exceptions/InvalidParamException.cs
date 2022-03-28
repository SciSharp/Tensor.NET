namespace Numnet.Exceptions{
    public class InvalidParamException:NNRuntimeException{
        public InvalidParamException(IntPtr status):base(status){}
    }
}