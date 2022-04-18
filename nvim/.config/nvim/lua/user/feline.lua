local is_status_ok, feline = pcall(require, 'feline')

if not is_status_ok then
  return
end

feline.setup{}
