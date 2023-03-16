import streamlit_authenticator as stauth

import database as db

usernames = ["rbances", "malcantara", "layamamani" , "ccabrera"]
names = ["Renzo Bances", "Manuel Alcantara", "Leibnithtz Ayamamani", "Cristian Cabrera"]
passwords = ["rb123", "ma123", "la123", "cc123"]
edades = [35, 36, 37 , 38]
generos = ["M", "M", "M", "M"]
pesos = [60.5, 65.5, 65.5, 65.5]
tallas = [1.65, 1.65, 1.65, 1.65]
imcs = [23.051, 23.051, 23.051, 23.051]
perabdominals = [63, 64, 65, 66]
cexweeks = [12, 12, 12, 12]
hashed_passwords = stauth.Hasher(passwords).generate()


for (username, name, hash_password, edad, genero, peso, talla, imc, perabdominal, cexweek) in zip(usernames, names, hashed_passwords, edades, generos, pesos, tallas, imcs, perabdominals, cexweeks):
    db.insert_user(username, name, hash_password, edad, genero, peso, talla, imc, perabdominal, cexweek)