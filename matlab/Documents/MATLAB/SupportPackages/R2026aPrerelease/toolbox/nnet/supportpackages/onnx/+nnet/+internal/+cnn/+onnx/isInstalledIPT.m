function tf = isInstalledIPT()

% Copyright 2020 The Mathworks, Inc.
tf = ~isempty(ver('images'));
end