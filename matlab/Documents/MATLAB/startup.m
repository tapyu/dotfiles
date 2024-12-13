%% Add path
if isunix
    xdg_data_home_path = getenv('XDG_DATA_HOME');
    % if `xdg_data_home_path` is empty, fallback to the default
    % `XDG_DATA_HOME` path
    if ~xdg_data_home_path
        xdg_data_home_path = [ '/home/' getenv('USER') '/.local/share' ];
    end

    matlab_path = [xdg_data_home_path '/matlab'];
    if isfolder(matlab_path)
        addpath(genpath(matlab_path));
        disp('Welcome to MATLAB! Custom paths have been added.');
    else
        error('There is no path %s to the added', matlab_path);
    end
    clear matlab_path xdg_data_home_path
elseif ispc
    error('There is no default paths for Windows.');
elseif ismac
    error('There is no default paths for macOS.');
else
    error('Unknown operating system.');
end

%% set schemer

% TODO: find out how to check if the current scheme if the defaut one
% if
%     [dir, ~, ~] = fileparts(which('schemer_import'));
%     schemer_import([dir '/schemes/darksteel.prf']);
% end