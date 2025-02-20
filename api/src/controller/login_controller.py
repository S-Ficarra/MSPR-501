from flask import request                                                           # type: ignore
from flask_restx import Namespace, Resource, fields                                 # type: ignore
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity  # type: ignore
from flask import jsonify                                                           # type: ignore
from connect_db import DBConnection
import bcrypt                                                                       # type: ignore


auth_namespace = Namespace('auth', description="Authentification des utilisateurs")

login_model = auth_namespace.model('Login', {
    'username': fields.String(required=True, description="Nom d'utilisateur"),
    'password': fields.String(required=True, description='Mot de passe')
})

token_model = auth_namespace.model('Token', {
    'access_token': fields.String(description="Token JWT d'authentification")
})

@auth_namespace.route('/login')
class LoginResource(Resource):
    @auth_namespace.expect(login_model)
    @auth_namespace.response(200, 'Succès', token_model)
    @auth_namespace.response(400, "Nom d'utilisateur et mot de passe requis")
    @auth_namespace.response(404, 'Utilisateur non trouvé')
    @auth_namespace.response(401, 'Mot de passe incorrect')
    @auth_namespace.response(500, 'Erreur serveur')
    def post(self):
        """Authentification et obtention d'un token JWT"""
        try:
            data = request.get_json()

            if not data or not isinstance(data, dict):
                return {'msg': "Données invalides"}, 400

            username = data.get('username')
            password = data.get('password')

            if not username or not password:
                return {'msg': "Nom d'utilisateur et mot de passe requis"}, 400

            with DBConnection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT password FROM users WHERE username = %s", (username,))
                user = cur.fetchone()

                if not user:
                    return {'msg': "Utilisateur non trouvé"}, 404

                hashed_password = user[0]

                if bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
                    access_token = create_access_token(identity=username)
                    return {'access_token': access_token}, 200
                else:
                    return {'msg': "Mot de passe incorrect"}, 401

        except Exception as e:
            return {'msg': f"Erreur serveur : {str(e)}"}, 500
