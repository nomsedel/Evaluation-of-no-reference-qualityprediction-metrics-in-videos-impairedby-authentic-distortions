% Modification made by:
% Jose Alejandro Ledesma and Stidl Alfonso torres
% Electronic Engineers 
% For graduate work "Evaluation of no-reference quality prediction metrics in videos impaired by authentic distortions"
% Pontificia Universidad Javeriana Cali, Santiago de Cali 2019-2020
% Supervised by:
% Hernán Darío Benítez Restrepo
% Roger Alfonso Gómez Nieto


function qualityscore  = brisquescore()

%ingresar la direccion de la carpeta donde estan los videos (esta linea debe terminar en \)
videos_path = 'D:\JAVERIANA\PRUEBAS\imagenes_para_comparacion\con_video\\';

%ingresar la direccion de la carpeta donde se guardaran .mat de las caracteristicas (esta linea debe terminar en \)
Feature_path ='D:\JAVERIANA\THESIS\RESULTS\BRISQUE\Qualcomm\Features\Colord\';

%ingresar la direccion de la carpeta donde se guardaran .mat de los scores (esta linea debe terminar en \)
Scorepath ='D:\JAVERIANA\THESIS\RESULTS\BRISQUE\Qualcomm\Scores\Colord\';

list=dir(videos_path)
for i=3:length(list)
i
list(i).name

% Se pasa la direccion del video a evaluar 
video = strcat(videos_path,list(i).name);    
obj = VideoReader(video);
vid = read(obj);
frames = obj.NumberOfFrames;
ST='.bmp';

% Se evalua cada frame del video independientemente, por lo que se pasan a formato bmp 
 for x = 1:1
      Sx=num2str(x);
      Strc=strcat(Sx,ST);
      imdist=vid(:,:,:,x);
if(size(imdist,3)==3)
    imdist = uint8(imdist);
    imdist = rgb2gray(imdist);
end
imdist = double(imdist);
if(nargin<2)
feat = brisque_feature(imdist);
end
fid = fopen('test_ind','w');
for jj = 1:size(feat,1)
fprintf(fid,'1 ');
for kk = 1:size(feat,2)
fprintf(fid,'%d:%f ',kk,feat(jj,kk));
end
fprintf(fid,'\n');
end
fclose(fid);
warning off all
delete output test_ind_scaled dump
system('svm-scale -r allrange test_ind >> test_ind_scaled');
system('svm-predict -b 1 test_ind_scaled allmodel output >>dump');
load output
Matriz_de_caracteristicas(x,:)=feat
qualityScore(x,1)=output;
 end

% Se guardan los resultados de los puntajes objetivos y caracteristicas en archivos Mat
 Fvideo=strcat(Feature_path,list(i).name);
 Fvideo2=strcat(Fvideo,'.mat');
 Svideo=strcat(Scorepath,list(i).name);
 Svideo2=strcat(Svideo,'.mat');
 save(Fvideo2,'Matriz_de_caracteristicas','-v7.3');  
 save (Svideo2,'qualityScore','-v7.3');
end