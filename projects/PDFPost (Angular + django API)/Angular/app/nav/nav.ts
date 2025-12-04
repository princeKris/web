import { Component, NgZone, OnInit } from '@angular/core';
import { SharedImports } from '../shared/shared-imports';
import { Auth } from '../services/auth';
import { Router } from '@angular/router';
import { ProfileService } from '../services/profile-service';

declare var bootstrap: any;
declare const google: any;
declare global {
  interface Window {
    google: any;
  }
}
@Component({
  selector: 'app-nav',
  imports: [...SharedImports],
  templateUrl: './nav.html',
  styleUrl: './nav.scss',
})
export class Nav implements OnInit  {
query: string = '';
susers: any[] = [];
sposts: any[] = [];
  isLoggedIn = false;
  user: any = null;
  pic="";
  loginUsername = '';
  loginPassword = '';
  signupUsername='';
  signupEmail='';
  signupPassword='';

  constructor(private auth: Auth, private router: Router,private profile:ProfileService,private ngZone: NgZone) {}

  ngOnInit(): void {
    this.auth.user$.subscribe(user => {
      this.user = user;
      this.isLoggedIn = !!user;
      console.log(user);
    });
    this.initGoogleSDK();

    document.addEventListener("click", (e: any) => {
      if (e.target?.classList?.contains("btn-close")) {
        try { e.target.blur(); } catch {}
      }
    });
  }

   /** ---------------------------
   * GOOGLE LOGIN INITIALIZER
   * --------------------------- */
    // Load Google script if not already there

 
 /** ----------------------------
   * GOOGLE LOGIN INITIALIZER
   * ---------------------------- */
  private initGoogleSDK() {
    const check = setInterval(() => {
      if (window['google']?.accounts?.id) {
        clearInterval(check);

        google.accounts.id.initialize({
          client_id:
            '809241562812-1jdnsdp9ornpei2q6lqjp11qd9hu220d.apps.googleusercontent.com',
          callback: (resp: any) =>
            this.ngZone.run(() =>
              this.handleGoogleCredential(resp.credential)
            ),
        });

        // Render buttons once SDK fully loaded
        this.renderGoogleButton('googleLoginBtn');
        this.renderGoogleButton('googleSignupBtn');
      }
    }, 100);
  }

  renderGoogleButton(id: string) {
    const el = document.getElementById(id);
    if (!el) return;

    google.accounts.id.renderButton(el, {
      theme: 'outline',
      size: 'large',
    });
  }

  handleGoogleCredential(id_token: string) {
    this.auth.googleBackendLogin(id_token).subscribe({
      next: () => {
        this.closeModal('signInModal');
      },
      error: () => alert('Google login failed'),
    });
  }

  submitLogin(): void {
    this.auth.login(this.loginUsername, this.loginPassword).subscribe({
      next: () => {
        //this.loginPassword='';
        alert("Login success");
        this.closeModal("signInModal");
      },
      error: () => alert("Invalid login")
    });
  }

  submitSignup(): void {
  const data = {
    username: this.signupUsername,
    email: this.signupEmail,
    password: this.signupPassword
  };

  console.log("Signup Form Data:", data);

  // Call your API
  this.auth.register(data).subscribe({
    next: () => {
      // this.signupEmail='';
      // this.signupPassword='';
      // this.signupUsername='';
      alert("Signup success!");
      this.closeModal('signUpModal');
    },
    error: () => {
      alert("Signup failed.");
    }
  });
}


  logout(): void {
    this.auth.logout();
    this.closeAllDropdowns();
    alert("you have been log out successfully")
  }

  closeAllDropdowns() {
    document.querySelectorAll('.dropdown-menu.show')
      .forEach((el: any) => el.classList.remove('show'));
  }

  closeModal(id: string) {
    const modalEl = document.getElementById(id);
    if (!modalEl) return;

    const active = document.activeElement as HTMLElement | null;
    if (active && modalEl.contains(active)) {
      try { active.blur(); } catch {}
    }

    const modal = bootstrap.Modal.getInstance(modalEl) || new bootstrap.Modal(modalEl);
    modal.hide();

    setTimeout(() => (document.body as HTMLElement).focus?.(), 200);
  }


  onSearch() {
  if (!this.query.trim()) {
    this.susers = [];
    this.sposts = [];
    return;
  }

  this.profile.search(this.query).subscribe({
    next: (res) => {
      this.susers = res.users;
      this.sposts = res.posts;
      console.log(this.sposts);
    },
    error: (err) => console.error(err)
  });
}

closeSidebar() {
  const sidebarEl = document.getElementById('mobileMenu');
  if (!sidebarEl) return;

  let offcanvas = bootstrap.Offcanvas.getInstance(sidebarEl);

  // If no instance exists, create it
  if (!offcanvas) {
    offcanvas = new bootstrap.Offcanvas(sidebarEl);
  }

  offcanvas.hide();
}




}
