-- create database jumps;
\connect jumps
drop table if exists jumper_vitals;
drop table if exists jumper;
drop table if exists category_connect;
drop table if exists category;
drop table if exists part_connect;
drop table if exists jump_part;

create table if not exists jumper(
uni_person varchar(4) primary key,
person_name varchar(32),
parent_contact_name varchar(64),
email varchar(255),
created_on TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
last_jump timestamp
);
create table if not exists jumper_vitals(
uni_person varchar(4) primary key, 
height float,
weight float, 
created_on TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
last_jump timestamp,
CONSTRAINT fk_jumper_jvitals
	FOREIGN KEY (uni_person) 
	REFERENCES jumper(uni_person)
);
create table if not exists category	(
uni_category serial primary key,
category_code varchar(8) unique not null,
category_name varchar(64),
created_on TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
updated_on TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
updated_by varchar(8)
);

create table if not exists category_connect	(
hi_level_num integer not null,
hi_category_code varchar(8)  not null,
hi_uni_name varchar(64),
lo_level_num integer not null, 
lo_category_code varchar(8)  not null,
lo_uni_name varchar(64),
created_on TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
updated_on TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
updated_by varchar(8),
CONSTRAINT fk_hi_category
	FOREIGN KEY(hi_category_code) 
	REFERENCES category(category_code),
CONSTRAINT fk_lo_category
	FOREIGN KEY(lo_category_code) 
	REFERENCES category(category_code)
);

CREATE UNIQUE INDEX category_connect_implode_idx ON category_connect
 (hi_level_num, hi_category_code);
CREATE UNIQUE INDEX category_connect_explode_idx ON category_connect
 (lo_level_num, lo_category_code);

create table if not exists jump_part(
uni_jump_part serial primary key,
jump_part_code varchar(8) unique not null,
jump_part_name varchar(64),
created_on TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
updated_on TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
updated_by varchar(8)
);

create table if not exists part_connect	(
hi_level_num integer not null,
hi_part_code varchar(8)  not null,
hi_part_name varchar(64),
lo_level_num integer not null, 
lo_part_code varchar(8)  not null,
lo_part_name varchar(64),
created_on TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
updated_on TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
updated_by varchar(8),
CONSTRAINT fk_hi_part
	FOREIGN KEY(hi_part_code) 
	REFERENCES jump_part(jump_part_code),
CONSTRAINT fk_lo_part
	FOREIGN KEY(lo_part_code) 
	REFERENCES jump_part(jump_part_code)
);

CREATE UNIQUE INDEX part_connect_implode_idx ON part_connect
 (hi_level_num, hi_part_code);

CREATE UNIQUE INDEX part_connect_explode_idx ON part_connect
 (lo_level_num, lo_part_code);