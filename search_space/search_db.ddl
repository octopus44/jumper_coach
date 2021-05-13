-- create database jumps;
\connect jumps
drop table if exists jumper_vitals;
drop table if exists jumper;
drop table if exists category_connect;
drop table if exists category;
drop table if exists part_connect;
drop table if exists jump_part;

create table if not exists jumper(
    uni_person serial primary key,
    person_code varchar(4) unique not null,
    person_name varchar(32),
    parent_contact_name varchar(64),
    email varchar(255),
    created_on TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_jump timestamp
);
create table if not exists jumper_vitals(
    person_code varchar(4) not null,
    height float,
    weight float,
    created_on TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_jump timestamp,
    PRIMARY KEY (person_code,created_on),
    CONSTRAINT fk_jumper_jvitals
        FOREIGN KEY (person_code)
        REFERENCES jumper(person_code)
);
create table if not exists category	(
    uni_category serial primary key,
    category_name varchar(64) unique not null,
    created_on TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_on TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_by varchar(8)
);

create table if not exists category_connect	(
    hi_level_num integer not null,
    hi_category_name varchar(64) not null,
    lo_level_num integer not null,
    lo_category_name varchar(64) not null,
    seq_num integer not null,
    created_on TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_on TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_by varchar(8),
    primary key (hi_level_num, hi_category_name, seq_num, lo_level_num, lo_category_name),
    CONSTRAINT fk_hi_category
        FOREIGN KEY(hi_category_name)
        REFERENCES category(category_name),
    CONSTRAINT fk_lo_category
        FOREIGN KEY(lo_category_name)
        REFERENCES category(category_name)
);

CREATE INDEX category_connect_implode_idx ON category_connect
 (hi_level_num, hi_category_name);
CREATE INDEX category_connect_explode_idx ON category_connect
 (lo_level_num, lo_category_name);

create table if not exists jump_part(
    uni_jump_part serial primary key,
    jump_part_name varchar(64) unique not null,
    created_on TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_on TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_by varchar(8)
);

create table if not exists part_connect	(
    hi_level_num integer not null,
    hi_part_name varchar(64) not null,
    seq_num integer not null,
    lo_level_num integer not null,
    lo_part_name varchar(64) not null,
    created_on TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_on TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_by varchar(8),
    primary key (hi_level_num, hi_part_name, seq_num, lo_level_num, lo_part_name),
    CONSTRAINT fk_hi_part
        FOREIGN KEY(hi_part_name)
        REFERENCES jump_part(jump_part_name),
    CONSTRAINT fk_lo_part
        FOREIGN KEY(lo_part_name)
        REFERENCES jump_part(jump_part_name)
);

CREATE INDEX part_connect_implode_idx ON part_connect
 (hi_level_num, hi_part_name);

CREATE INDEX part_connect_explode_idx ON part_connect
 (lo_level_num, lo_part_name);