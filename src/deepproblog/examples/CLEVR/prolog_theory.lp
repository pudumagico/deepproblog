:-style_check(-discontiguous).
:-style_check(-singleton).
:- dynamic unique/2.
:- dynamic relate_left/2.
:- dynamic relate_right/2.
:- dynamic relate_front/2.
:- dynamic relate_behind/2.
:- dynamic count/2.
:- dynamic exist/2.
:- dynamic filter_large/2.
:- dynamic filter_small/2.
:- dynamic filter_gray/2.
:- dynamic filter_red/2.
:- dynamic filter_blue/2.
:- dynamic filter_green/2.
:- dynamic filter_brown/2.
:- dynamic filter_purple/2.
:- dynamic filter_cyan/2.
:- dynamic filter_yellow/2.
:- dynamic filter_metal/2.
:- dynamic filter_rubber/2.
:- dynamic filter_sphere/2.
:- dynamic filter_cylinder/2.
:- dynamic filter_cube/2.
:- dynamic query_size/2.
:- dynamic query_color/2.
:- dynamic query_material/2.
:- dynamic query_shape/2.
:- dynamic and/3.
:- dynamic or/2.
:- dynamic or/3.
:- dynamic same_size/2.
:- dynamic same_color/2.
:- dynamic same_material/2.
:- dynamic same_shape/2.
:- dynamic equal_integer/3.
:- dynamic less_than/3.
:- dynamic greater_than/3.
:- dynamic equal_size/3.
:- dynamic equal_color/3.
:- dynamic equal_material/3.
:- dynamic equal_shape/3.
:- dynamic size/2.
:- dynamic color/2.
:- dynamic material/2.
:- dynamic shape/2.
:- dynamic bool/2.
:- dynamic int/2.
:- dynamic label/4.

% Scene rule
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- scene(T,0), Z is T+1, obj(0,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).

% Unique rule/constraint
% obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- unique(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).
%:- unique(T,T1), obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(T1,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21), ID=\=ID1.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- unique(T,T1), Z is T+1, obj(T,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(T1,ID1,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), ID \== ID1,!, fail.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- unique(T,T1), Z is T+1, obj(T,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).


% Relate rules
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- relate_left(T,T1), Z is T+1, obj(0,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(T1,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21), ID=\=ID1, X1<X11.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- relate_right(T,T1), Z is T+1, obj(0,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(T1,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21), ID=\=ID1, X1>=X11.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- relate_front(T,T1), Z is T+1, obj(0,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(T1,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21), ID=\=ID1, Y2>Y21.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- relate_behind(T,T1), Z is T+1, obj(0,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(T1,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21), ID=\=ID1, Y2=<Y21.

% Count rule
int(T,N) :- count(T,T1), aggregate_all(count, ID, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), N).

% Exist rule
bool(T,true) :- exist(T,T1), obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).
bool(T,false) :- exist(T,T1), \+ bool(T,true).

% Filtering rules
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_large(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), SIZE=large.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_small(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), SIZE=small.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_gray(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), COLOR=gray.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_red(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), COLOR=red.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_blue(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), COLOR=blue.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_green(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), COLOR=green.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_brown(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), COLOR=brown.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_purple(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), COLOR=purple.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_cyan(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), COLOR=cyan.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_yellow(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), COLOR=yellow.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_metal(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), MATERIAL=metal.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_rubber(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), MATERIAL=rubber.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_sphere(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), SHAPE=sphere.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_cylinder(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), SHAPE=cylinder.
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- filter_cube(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), SHAPE=cube.

% Query functions
size(T,SIZE) :- query_size(T,T1), obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).
color(T,COLOR) :- query_color(T,T1), obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).
material(T,MATERIAL) :- query_material(T,T1), obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).
shape(T,SHAPE) :- query_shape(T,T1), obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).

% Logical operators
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- and(T,T1,T2), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(T2,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).

or(T,T1) :- or(T,T1,T2).
or(T,T2) :- or(T,T1,T2).
obj(Z,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- or(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2).

% Same-attribute relations
obj(Z,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21) :- same_size(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(0,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21), ID=\=ID1, SIZE=SIZE1.
obj(Z,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21) :- same_color(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(0,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21), ID=\=ID1, COLOR=COLOR1.
obj(Z,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21) :- same_material(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(0,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21), ID=\=ID1, MATERIAL=MATERIAL1.
obj(Z,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21) :- same_shape(T,T1), Z is T+1, obj(T1,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2), obj(0,ID1,SIZE1,COLOR1,MATERIAL1,SHAPE1,X11,Y11,X21,Y21), ID=\=ID1, SHAPE=SHAPE1.

% Integer comparison
bool(T,true) :- equal_integer(T,T1,T2), int(T1,V1), int(T2,V2), V1=V2.
bool(T,false) :- equal_integer(T,T1,T2), \+ bool(T,true).

bool(T,true) :- less_than(T,T1,T2), int(T1,V1), int(T2,V2), V1<V2.
bool(T,false) :- less_than(T,T1,T2), \+ bool(T,true).

bool(T,true) :- greater_than(T,T1,T2), int(T1,V1), int(T2,V2), V1>V2.
bool(T,false) :- greater_than(T,T1,T2), \+ bool(T,true).

% Attribute comparison
bool(T,true) :- equal_size(T,T1,T2), size(T1,V1), size(T2,V2), V1=V2.
bool(T,false) :- equal_size(T,T1,T2), \+ bool(T,true).

bool(T,true) :- equal_color(T,T1,T2), color(T1,V1), color(T2,V2), V1=V2.
bool(T,false) :- equal_color(T,T1,T2), \+ bool(T,true).

bool(T,true) :- equal_material(T,T1,T2), material(T1,V1), material(T2,V2), V1=V2.
bool(T,false) :- equal_material(T,T1,T2), \+ bool(T,true).

bool(T,true) :- equal_shape(T,T1,T2), shape(T1,V1), shape(T2,V2), V1=V2.
bool(T,false) :- equal_shape(T,T1,T2), \+ bool(T,true).

% Derive answer (T must equal the last point in time)
ans(V) :- end(T), size(T,V).
ans(V) :- end(T), color(T,V).
ans(V) :- end(T), material(T,V).
ans(V) :- end(T), shape(T,V).
ans(V) :- end(T), bool(T,V).
ans(V) :- end(T), int(T,V).

obj(0,ID,SIZE,COLOR,MATERIAL,SHAPE,X1,Y1,X2,Y2) :- label(0,img,ID,obj(ID,SHAPE,SIZE,COLOR,MATERIAL,X1,Y1,X2,Y2)).

