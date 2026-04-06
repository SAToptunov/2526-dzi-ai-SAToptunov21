from flask import render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from . import auth_bp
from ..models import db, User  # Използваме .. за да вземем models от горната папка


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            flash('Потребителското име вече е заето.', 'danger')
            return redirect(url_for('auth.register'))

        new_user = User(username=username)
        new_user.set_password(password)
        if username.lower() == 'admin':
            new_user.role = 'admin'

        db.session.add(new_user)
        db.session.commit()
        flash('Успешна регистрация!', 'success')
        return redirect(url_for('auth.login'))
    return render_template('register.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))

    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and user.check_password(request.form.get('password')):
            login_user(user)
            return redirect(url_for('main.dashboard'))
        flash('Грешни данни.', 'danger')
    return render_template('login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.index'))