import { Routes } from '@angular/router';
import { loginGuard } from './guards/login-guard';

export const routes: Routes = [
    {path:'',loadComponent:()=>import('./home/home').then(m=>m.Home)},
    {path:'home',loadComponent:()=>import('./home/home').then(m=>m.Home)},
    {path:'profile',canActivate:[loginGuard],loadComponent:()=>import('./user-profile/user-profile').then(m=>m.UserProfile)},
    {path:'profile/:id',loadComponent:()=>import('./viwe-profile/viwe-profile').then(m=>m.ViweProfile)},
    {path:'post/:id',loadComponent:()=>import('./view-pdfpost/view-pdfpost').then(m=>m.ViewPdfpost)},
    {path:'**',loadComponent:()=>import('./create-pdfpost/create-pdfpost').then(m=>m.CreatePdfpost)}
];
