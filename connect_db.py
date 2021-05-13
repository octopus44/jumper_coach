#!/usr/bin/python
import psycopg2
import os

def store_jumpers():
    count = 0
    batchsql = ''
    with open('search_space/insert_ref_data.sql', 'r') as fm:
        Lines = fm.readlines()
        for line in Lines:
            count += 1
            batchsql = batchsql + " " + line.strip()
    return batchsql

def exec_statement(conn, stmt):
    try:
        db_crsr = conn.cursor()
        db_crsr.execute(stmt)
        rowcount = db_crsr.rowcount
        print(rowcount)
        conn.commit()
        db_crsr.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')


def connect_close(conn, cur):
        # close the communication with the PostgreSQL
    try:
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')


def connect_open():
    """ Connect to the PostgreSQL database server """
    conn = None
    batchquery = ''
    try:
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect( host = "localhost",
                    database = "jumps",
                    user = "postgres",
                    password = os.environ['DBPASS'])

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')
    return conn


def connect():
    """ Connect to the PostgreSQL database server """
    conn = None
    batchquery = ''
    try:
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect( host = "localhost",
                    database = "jumps",
                    user = "postgres",
                    password = os.environ['DBPASS'])

        # create a cursor
        cur = conn.cursor()

        # execute a statement
        print('PostgreSQL database version:')

        batchquery  = store_jumpers()
        cur.execute(batchquery)
        rowcount = cur.rowcount
        print(rowcount)
        conn.commit()
        cur.execute('SELECT version()')
        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        print(db_version)

        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')


if __name__ == '__main__':
    connect()
