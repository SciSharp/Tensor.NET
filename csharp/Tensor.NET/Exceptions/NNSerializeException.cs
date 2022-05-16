using Tensornet.Native;

namespace Tensornet.Exceptions{
    public class NNSerializeException:Exception{
        protected string _message;
        public override string Message => _message;
        public NNSerializeException(string message){
            _message = message;
        }
    }
}