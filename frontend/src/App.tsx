import { Amplify, Auth } from "aws-amplify";
import { withAuthenticator } from "@aws-amplify/ui-react";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "./index.css";
import Layout from "./routes/layout";
import Documents from "./routes/documents";
import Chat from "./routes/chat";

Amplify.configure({
  Auth: {
    userPoolId: import.meta.env.VITE_USER_POOL_ID,
    userPoolWebClientId: import.meta.env.VITE_USER_POOL_CLIENT_ID,
    region: import.meta.env.VITE_API_REGION,
  },
  API: {
    endpoints: [
      {
        name: "serverless-pdf-chat",
        endpoint: import.meta.env.VITE_API_ENDPOINT,
        region: import.meta.env.VITE_API_REGION,
        custom_header: async () => {
          return {
            Authorization: `Bearer ${(await Auth.currentSession())
              .getIdToken()
              .getJwtToken()}`,
          };
        },
      },
    ],
  },
});

let router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      {
        index: true,
        Component: Documents,
      },
      {
        path: "/doc/:documentid/:conversationid",
        Component: Chat,
      },
    ],
  },
]);

function App() {
  return <RouterProvider router={router} />;
}


// export function AuthStyle() {
//   const { tokens } = useTheme();
//   const theme: Theme = {
//     name: 'Auth Example Theme',
//     tokens: {
//       components: {
//         authenticator: {
//           router: {
//             boxShadow: `0 0 16px ${tokens.colors.overlay['10']}`,
//             borderWidth: '0',
//           },
//           form: {
//             padding: `${tokens.space.medium} ${tokens.space.xl} ${tokens.space.medium}`,
//           },
//         },
//         button: {
//           primary: {
//             backgroundColor: tokens.colors.neutral['100'],
//           },
//           link: {
//             color: tokens.colors.purple['80'],
//           },
//         },
//         fieldcontrol: {
//           _focus: {
//             boxShadow: `0 0 0 2px ${tokens.colors.purple['60']}`,
//           },
//         },
//         tabs: {
//           item: {
//             color: tokens.colors.neutral['80'],
//             _active: {
//               borderColor: tokens.colors.neutral['100'],
//               color: tokens.colors.purple['100'],
//             },
//           },
//         },
//       },
//     },
//   };

//   return (
//     <ThemeProvider theme={theme}>
//       <View padding="xxl">
//         <Authenticator />
//       </View>
//     </ThemeProvider>
//   );
// }

export default withAuthenticator(App, { hideSignUp: true});
