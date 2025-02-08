CREATE TABLE region(
   id_region SERIAL,
   name VARCHAR(50) NOT NULL,
   population BIGINT, -- Adapté pour supporter de grandes populations
   PRIMARY KEY(id_region)
);

CREATE TABLE continent(
   id_continent SERIAL,
   name VARCHAR(50) NOT NULL,
   PRIMARY KEY(id_continent)
);

CREATE TABLE climat_type(
   id_climat_type SERIAL,
   name VARCHAR(50) NOT NULL,
   description VARCHAR(50),
   PRIMARY KEY(id_climat_type)
);

CREATE TABLE country(
   id_country SERIAL,
   name VARCHAR(50) NOT NULL,
   population BIGINT NOT NULL, -- Adapté pour de grandes populations
   pib NUMERIC(18,2), -- Plus précis que MONEY
   latitude DOUBLE PRECISION,
   longitude DOUBLE PRECISION,
   id_climat_type INTEGER NOT NULL,
   id_continent INTEGER NOT NULL,
   id_region INTEGER,
   PRIMARY KEY(id_country),
   FOREIGN KEY(id_climat_type) REFERENCES climat_type(id_climat_type),
   FOREIGN KEY(id_continent) REFERENCES continent(id_continent),
   FOREIGN KEY(id_region) REFERENCES region(id_region)
);

CREATE TABLE weather_report(
   id_weather_report SERIAL,
   date_start DATE NOT NULL,
   date_end DATE NOT NULL,
   average_temperature NUMERIC(15,2),
   average_wind_velocity NUMERIC(15,2),
   humidity_level NUMERIC(15,2),
   id_country INTEGER NOT NULL,
   PRIMARY KEY(id_weather_report),
   FOREIGN KEY(id_country) REFERENCES country(id_country)
);

CREATE TABLE disease(
   id_disease SERIAL,
   name VARCHAR(50) NOT NULL,
   is_pandemic BOOLEAN NOT NULL,
   PRIMARY KEY(id_disease)
);

CREATE TABLE statement(
   id_statement SERIAL,
   _date DATE NOT NULL,
   confirmed NUMERIC(20,0) NOT NULL,
   deaths NUMERIC(20,0) NOT NULL,
   recovered NUMERIC(20,0) NOT NULL,
   active NUMERIC(20,0) NOT NULL,
   total_tests NUMERIC(20,0),
   new_deaths NUMERIC(20,0) DEFAULT 0,
   new_cases NUMERIC(20,0) DEFAULT 0,
   new_recovered NUMERIC(20,0) DEFAULT 0,
   id_disease INTEGER NOT NULL,
   id_country INTEGER,
   PRIMARY KEY(id_statement),
   FOREIGN KEY(id_disease) REFERENCES disease(id_disease),
   FOREIGN KEY(id_country) REFERENCES country(id_country)
);









