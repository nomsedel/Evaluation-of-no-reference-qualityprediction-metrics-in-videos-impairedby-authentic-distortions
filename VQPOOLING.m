% Jose Alejandro Ledesma and Stidl Alfonso torres
% Electronic Engineers 
% For graduate work "Evaluation of no-reference quality prediction metrics in videos impaired by authentic distortions"
% Pontificia Universidad Javeriana Cali, Santiago de Cali 2019-2020
% Supervised by:
% Hernán Darío Benítez Restrepo
% Roger Alfonso Gómez Nieto

% Se pasa la dirrecion de la carpeta con los puntajes objetivos extraidos 
videos_path = 'D:\JAVERIANA\THESIS\RESULTS\NIQE\REPETICION\CVD\TEST6\TELEVISION\';
list=dir(videos_path)
k=1;
for m=3:length(list) 
    video_name{k,1}=list(m).name;
    video=strcat(videos_path,list(m).name);
    Scores_videos=load(video);

qualityScore = Scores_videos.qualityScore;
average_pooling=mean(qualityScore);

% Se dividen los puntajes en dos grupos 
[idx,C]= kmeans(qualityScore,2);
for i=1:length(qualityScore)
% Se asigna cada puntaje objetivo a alguno de los grupos 
    if idx(i,1) == 1
        M1(i)= qualityScore(i);
    else 
        M2(i)= qualityScore(i);
    end 
end 

% Se eliminan los ceros que deja la asignacion anterior 
m1=M1(find(M1~=0)) ;   
m2=M2(find(M2~=0));

Max1=max(m1);
Max2=max(m2);
suma1=sum(m1);
suma2=sum(m2);
c1=length(m1);
c2=length(m2);
a1=mean(m1);
a2=mean(m2);

% Se realizan las ecuaciones pertinentes presentadas en el trabajo de grado 
if Max1>Max2
    W=(1-(a2/a1));
    VQPooling= (suma1+W*suma2)/(c2+W*c1);
else  
    W=(1-(a1/a2));
    VQPooling= (suma2+W*suma1)/(c1+W*c2);
end 
matriz_average_pooling(k,1)=average_pooling;
matriz_VQPooling(k,1)=VQPooling;
k=k+1;
end 
