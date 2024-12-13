% Display a custom message
disp('Welcome to MATLAB! Custom paths have been added.');

%% Add path
if isunix
    xdg_data_home_path = getenv('XDG_DATA_HOME');
    % if `xdg_data_home_path` is empty, the default `XDG_DATA_HOME` path
    if ~xdg_data_home_path
        xdg_data_home_path = [ '/home/' getenv('USER') '/.local/share' ];
    end

    matlab_path = [xdg_data_home_path '/matlab'];
    if isfolder(matlab_path)
        addpath(genpath(matlab_path));
    else
        error('There is no path %s to the added', matlab_path);
    end
elseif ispc
    error('There is no default paths for Windows.');
elseif ismac
    error('There is no default paths for macOS.');
else
    error('Unknown operating system.');
end
