using Numnet.Native;

namespace Numnet.Exceptions{
    public class NNRuntimeException:Exception{
        protected string _message;
        public override string Message => _message;
        public NNRuntimeException(string message){
            _message = message;
        }
    }
}