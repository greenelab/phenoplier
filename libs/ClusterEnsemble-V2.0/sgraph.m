% function labels = sgraph(k,dataname)
%
% copyright (c) 1998-2011 by Alexander Strehl

function labels = sgraph(k,dataname) 

scripts_dir = getenv('SCRIPTS_DIR');
if isempty(scripts_dir)
    scripts_dir = tempdir();
end;

temp_script_file = tempname(scripts_dir);

pmetis_path = getenv('PMETIS_PATH');
if isempty(pmetis_path)
    pmetis_path = 'pmetis';
end;
if ismac()
    pmetis_path = [pmetis_path '.macos']
end;

shmetis_path = getenv('SHMETIS_PATH');
if isempty(shmetis_path)
    shmetis_path = 'shmetis';
end;
if ismac()
    shmetis_path = [shmetis_path '.macos']
end;

script_ext = '.sh';
if ~isunix()
    script_ext = '.bat';
end;

scriptfile = [temp_script_file 'partgraph' num2str(sum(dataname=='0')) num2str(sum(dataname=='1')) num2str(sum(dataname=='2')) num2str(sum(dataname=='3')) script_ext];

if ~exist('dataname'),
      dataname = [temp_script_file '' 'graph0'];
end;
resultname = [dataname '.part.' num2str(k)];

lastchar = str2num(dataname(length(dataname)));
if (isempty(lastchar)),
  disp('sgraph: file does not comply to name convention');
  lastchar = 0;
end;
fid = fopen(scriptfile,'w');

if isunix()
   fprintf(fid,'%s\n', '#!/bin/sh');
end;

if (lastchar<2),
   fprintf(fid,'%s\n',[pmetis_path ' ' dataname ' ' num2str(k)]);
else
   ubfactor = 5;
   fprintf(fid,'%s\n',[shmetis_path ' ' dataname ' ' num2str(k) ' ' num2str(ubfactor)]);
end;
fclose(fid);

if isunix()
    system(['chmod +x ' scriptfile]);
end;

system(scriptfile);

delete(scriptfile);

fid = fopen(resultname,'r');
if (fid == -1),
  disp('sgraph: partitioning not successful due to external error');
  fid = fopen(dataname);
  if (fid == -1),
    disp('sgraph: graph file is not accessible');
  else
    if lastchar>=2,
      junk = fscanf(fid,'%d',1); 
    end;
    labels = ones(1,fscanf(fid,'%d',1));
    if isempty(labels),
      disp('sgraph: empty label vector - suspecting file system full');
    end;
    fclose(fid);
  end;
else
  disp(['sgraph: ' scriptfile ' completed - loading ' resultname]);
  labels = (fscanf(fid,'%d')+1)';
  fclose(fid);
end;

fid = fopen(resultname,'r');
if (fid ~= -1),
  fclose(fid);
  delete(resultname);
end;
