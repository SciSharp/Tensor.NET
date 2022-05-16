namespace Tensornet.Exceptions{
    public class MismatchedTypeException:Exception{
        protected string _message;
        public override string Message => _message;
        public MismatchedTypeException(string message){
            _message = message;
        }
        public MismatchedTypeException(){
            _message = "";
        }
    }
}