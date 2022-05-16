using Tensornet.Native;

namespace Tensornet.Exceptions{
    public class NNRuntimeException:Exception{
        protected string _message;
        public override string Message => _message;
        public NNRuntimeException(string message){
            _message = message;
        }
    }
}