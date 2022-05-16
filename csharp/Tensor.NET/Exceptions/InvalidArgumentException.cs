namespace Tensornet.Exceptions{
    public class InvalidArgumentException:Exception{
        protected string _message;
        public override string Message => _message;
        public InvalidArgumentException(string info){
            _message = info;
        }
    }
}