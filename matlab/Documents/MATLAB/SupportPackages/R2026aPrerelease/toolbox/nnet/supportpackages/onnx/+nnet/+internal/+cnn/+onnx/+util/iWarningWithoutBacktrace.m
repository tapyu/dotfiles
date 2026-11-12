function iWarningWithoutBacktrace(msgID, varargin)
backtrace = warning('query','backtrace');
warning('off','backtrace');
warning(message(msgID, varargin{:}));
warning(backtrace.state,'backtrace');
end